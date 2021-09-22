import threading

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from six.moves.queue import Queue
import utils.some_loss as some_loss
import utils.some_trace as some_trace
import gym

ray.init()
env_fn = lambda: gym.make('LunarLander-v2')


class TestPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.hidden_dim = 128
        self.layer = torch.nn.LSTM(input_size=128, hidden_size=128, num_layers=1)
        self.action = nn.Sequential(torch.nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 4))
        self.v = nn.Sequential(torch.nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x, h=None):
        x = self.pre(x)
        x, h = self.layer(x, h)
        return self.action(x), h, self.v(x)

    def learn_from_batch(self, batch: dict):
        s = batch['s']  # .permute(1, 0, 2)
        a = batch['a']  # .permute(1, 0, 2)
        h = batch['init_h']
        notd = ~(batch['d'])  # .permute(1, 0, 2))
        T, B, *_ = a.shape
        core_output = self.pre(s.reshape(T * B, -1)).reshape(T, B, -1)
        core_output_list = []
        for xx, nd in zip(core_output.unbind(), notd.unbind()):
            nd = nd.view(1, -1, 1)
            h = tuple(nd * s for s in h)
            output, h = self.layer(xx.unsqueeze(0), h)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        new_logp = self.action(core_output)
        dist = torch.distributions.Categorical(new_logp.softmax(dim=-1))
        a = dist.sample().reshape(T, B)
        v = self.v(core_output).reshape(T, B)
        return {
            'v': v,
            'logp': new_logp.reshape(T, B, -1),
            'a': a
        }

    @torch.no_grad()
    def act(self, x, h):
        x, h, v = self.forward(x, h)
        logp = x
        x = torch.distributions.Categorical(x.softmax(dim=-1))
        a = x.sample()
        return a, v, logp, h

    def get_parameter(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}


@ray.remote
class Rollout(threading.Thread):
    def __init__(self, model_fn, env_fn, observation_shape, T):
        super().__init__()
        self.env: gym.Env = env_fn()
        self.model: torch.nn.Module = model_fn()
        self.T = T
        self.running = False
        self.queue = Queue(maxsize=50)
        self.lock = threading.Lock()
        self.observation_shape = observation_shape

    def run(self):
        self.running = True
        s = self.env.reset()
        re = 0
        while self.running:
            batch = {
                's': torch.zeros([self.T, *self.observation_shape]),
                'a': torch.zeros([self.T]).long(),
                'r': torch.zeros([self.T]),
                'd': torch.zeros([self.T, ]).bool(),
                'logp': torch.zeros([self.T, self.env.action_space.n]),
                'v': torch.zeros([self.T, ]),
                'return': torch.zeros([self.T, ]),
                'init_h': (torch.zeros([1, 1, self.model.hidden_dim]), torch.zeros([1, 1, self.model.hidden_dim]))
            }
            h = (torch.zeros([1, 1, self.model.hidden_dim]), torch.zeros([1, 1, self.model.hidden_dim]))
            for idx in range(self.T):
                batch['s'][idx] = torch.FloatTensor(s)
                if idx == 0:
                    batch['init_h'] = h
                a, v, logp, h = self.model.act(torch.FloatTensor(s).unsqueeze(0).unsqueeze(0), h)
                batch['a'][idx] = a.clone()
                s, r, d, info = self.env.step(a.item())
                re += r
                batch['r'][idx] = torch.FloatTensor([r]) / 200.0
                batch['d'][idx] = torch.BoolTensor([d])
                batch['v'][idx] = v.clone()
                batch['logp'][idx] = logp.clone()
                if d:
                    batch['return'][idx] = torch.FloatTensor([re])
                    re = 0
                    s = self.env.reset()
            self.queue.put(batch)

    def get_data(self):
        if not self.is_alive():
            raise RuntimeError("thread has died")
        return self.queue.get(timeout=600)

    def set_parameter(self, parameter):
        with self.lock:
            self.model.load_state_dict(parameter)

    def set_running(self, running):
        self.running = running


class Batcher(threading.Thread):
    def __init__(self, actor, device, queue):
        super(Batcher, self).__init__()
        self.actors = actor
        self.queue = queue
        self.device = device

    def run(self) -> None:
        while True:
            data: dict = ray.get(self.actors.get_data.remote())
            v: torch.Tensor = torch.zeros([1])
            v.to(device=self.device, non_blocking=True)
            data = {k: v.to(device=self.device, non_blocking=True) if isinstance(v, torch.Tensor) else tuple(
                i.to(device=self.device, non_blocking=True) for i in v) for k, v in data.items()}
            self.queue.put(data)


class Learner(threading.Thread):
    def __init__(self, model_fn, env_fn, device, batch_size, actors):
        super().__init__()
        self.env = env_fn()
        self.model: nn.Module = model_fn()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.running = False
        self.device = device
        self.model.to(self.device)
        self.queue = Queue(maxsize=5000)
        self.ok_queue = Queue(maxsize=5)
        self.lock = threading.Lock()
        self.action = 0
        self.batch_size = batch_size
        self.actors = actors
        self.worker = [Batcher(actor, self.device, self.queue) for actor in actors]

        weight = ray.put(self.model.get_parameter())  # .remote()
        ray.get([actor.set_parameter.remote(weight) for actor in actors])
        for work in self.worker:
            work.start()


    def get_batch(self, batch_size):
        if not self.is_alive():
            raise RuntimeError("thread has died")
        datas = []
        for i in range(batch_size):
            datas.append(self.queue.get(timeout=600))
        new_datas = {}
        for key in datas[0].keys():
            if key != 'init_h':
                new_datas[key] = torch.stack([i[key] for i in datas]).transpose(1, 0)
        if 'init_h' in datas[0].keys():
            init_h = (
                torch.cat(ts, dim=1)
                for ts in zip(*[i['init_h'] for i in datas])
            )
            new_datas['init_h'] = tuple(
                t.to(device=self.device, non_blocking=True) for t in init_h
            )
        return new_datas

    def run(self):
        while True:
            datas = self.get_batch(self.batch_size)

            # for k, v in datas.items():
            #     print(k, v)
            # self.model.learn()
            stat = self.learn_from_batch(datas)
            print(stat)
            weight = ray.put(self.model.get_parameter())
            [actor.set_parameter.remote(weight) for actor in actors]
            # ray.get([actor.set_parameter.remote(self.model.get_parameter()) for actor in self.actors])
            import time
            # time.sleep(3)

    def learn_from_batch(self, batch):
        learner_outputs = self.model.learn_from_batch(batch)
        bootstrap_value = learner_outputs["v"][-1]
        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items() if key != 'init_h'}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        rewards = batch["r"].clip(-1, 1)

        discounts = (~batch["d"]).float() * 1.0

        vtrace_returns = some_trace.v_trace_from_logits(
            behavior_policy_logits=batch["logp"],
            target_policy_logits=learner_outputs["logp"],
            actions=batch["a"],
            discounts=discounts,
            rewards=rewards,
            values=learner_outputs["v"],
            bootstrap_value=bootstrap_value,
        )

        # print('retrace ok')
        pg_loss = some_loss.compute_policy_gradient_loss(
            learner_outputs["logp"],
            batch["a"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = 0.5 * some_loss.compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["v"]
        )
        entropy_loss = 1e-3 * some_loss.compute_entropy_loss(
            learner_outputs["logp"]
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch["return"][batch["d"]]

        self.optim.zero_grad()
        total_loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 20)
        self.optim.step()
        return {
            'pg_loss': pg_loss.item(),
            'v_loss': baseline_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'return': episode_returns.mean().item(),
            'v_mean': vtrace_returns.vs.mean().item(),
            'r_mean': rewards.mean().item(),
            'grad': grad.norm().item()
        }

    def set_parameter(self, parameter):
        with self.lock:
            self.action = parameter

    def set_running(self, running):
        self.running = running


model_fn = TestPolicy
actors = []
for _ in range(32):
    actor = Rollout.remote(model_fn=model_fn, env_fn=env_fn, observation_shape=[8], T=50)
    actor.start.remote()
    actors.append(actor)

learner = Learner(model_fn=model_fn, env_fn=env_fn, device='cuda:1', batch_size=256, actors=actors)
learner.start()
#         s = self.env.reset()
#         while self.running:
#             a = self.model.act(s)
#             s_next, r, d, info = self.env.step(a)
#             if d:
#                 s_next = self.env.reset()
#         try:
#             logging.info("Actor %i started.", actor_index)
#             timings = prof.Timings()  # Keep track of how fast things are.
#
#             gym_env = create_env(flags)
#             seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
#             gym_env.seed(seed)
#             env = environment.Environment(gym_env)
#             env_output = env.initial()
#             agent_state = model.initial_state(batch_size=1)
#             agent_output, unused_state = model(env_output, agent_state)
#             while True:
#                 data_queue.
#                 index = free_queue.get()
#                 if index is None:
#                     break
#
#                 # Write old rollout end.
#                 for key in env_output:
#                     buffers[key][index][0, ...] = env_output[key]
#                 for key in agent_output:
#                     buffers[key][index][0, ...] = agent_output[key]
#                 for i, tensor in enumerate(agent_state):
#                     initial_agent_state_buffers[index][i][...] = tensor
#
#                 # Do new rollout.
#                 env_step_time =0
#                 model_infer_time=0
#                 model_infer_time_total=0
#                 for t in range(flags.unroll_length):
#                     timings.reset()
#                     tt =time.time()
#                     with torch.no_grad():
#                         agent_output, agent_state = model(env_output, agent_state)
#                     model_infer_time=max(time.time()-tt,model_infer_time)
#                     model_infer_time_total+=time.time()-tt
#                     timings.time("model")
#                     tt = time.time()
#                     env_output = env.step(agent_output["action"])
#                     env_step_time+=time.time()-tt
#
#                     timings.time("step")
#
#                     for key in env_output:
#                         buffers[key][index][t + 1, ...] = env_output[key]
#                     for key in agent_output:
#                         buffers[key][index][t + 1, ...] = agent_output[key]
#
#                     timings.time("write")
#                 print(actor_index,'env model',model_infer_time_total,model_infer_time)
#                 full_queue.put(index)
#
#                 if actor_index == 0:
#                     logging.info("Actor %i: %s", actor_index, timings.summary())
#
#         except KeyboardInterrupt:
#             pass  # Return silently.
#         except Exception as e:
#             logging.error("Exception in worker process %i", actor_index)
#             traceback.print_exc()
#             print()
#             raise e
# import threading
# from six.moves import queue
# class LearnerThread(threading.Thread):
#     """Background thread that updates the local model from sample trajectories.
#     The learner thread communicates with the main thread through Queues. This
#     is needed since Ray operations can only be run on the main thread. In
#     addition, moving heavyweight gradient ops session runs off the main thread
#     improves overall throughput.
#     """
#
#     def __init__(self, local_worker: RolloutWorker, minibatch_buffer_size: int,
#                  num_sgd_iter: int, learner_queue_size: int,
#                  learner_queue_timeout: int):
#         """Initialize the learner thread.
#         Args:
#             local_worker (RolloutWorker): process local rollout worker holding
#                 policies this thread will call learn_on_batch() on
#             minibatch_buffer_size (int): max number of train batches to store
#                 in the minibatching buffer
#             num_sgd_iter (int): number of passes to learn on per train batch
#             learner_queue_size (int): max size of queue of inbound
#                 train batches to this thread
#             learner_queue_timeout (int): raise an exception if the queue has
#                 been empty for this long in seconds
#         """
#         threading.Thread.__init__(self)
#         self.learner_queue_size = WindowStat("size", 50)
#         self.local_worker = local_worker
#         self.inqueue = queue.Queue(maxsize=learner_queue_size)
#         self.outqueue = queue.Queue()
#         self.minibatch_buffer = MinibatchBuffer(
#             inqueue=self.inqueue,
#             size=minibatch_buffer_size,
#             timeout=learner_queue_timeout,
#             num_passes=num_sgd_iter,
#             init_num_passes=num_sgd_iter)
#         self.queue_timer = TimerStat()
#         self.grad_timer = TimerStat()
#         self.load_timer = TimerStat()
#         self.load_wait_timer = TimerStat()
#         self.daemon = True
#         self.weights_updated = False
#         self.stats = {}
#         self.stopped = False
#         self.num_steps = 0
#
#     def run(self) -> None:
#         while not self.stopped:
#             self.step()
#
#     def step(self) -> Optional[_NextValueNotReady]:
#         with self.queue_timer:
#             try:
#                 batch, _ = self.minibatch_buffer.get()
#             except queue.Empty:
#                 return _NextValueNotReady()
#
#         with self.grad_timer:
#             fetches = self.local_worker.learn_on_batch(batch)
#             self.weights_updated = True
#             self.stats = get_learner_stats(fetches)
#
#         self.num_steps += 1
#         self.outqueue.put((batch.count, self.stats))
#         # self.learner_queue_size.push(self.inqueue.qsize())
#
#     def add_learner_metrics(self, result: Dict) -> Dict:
#         """Add internal metrics to a trainer result dict."""
#
#         def timer_to_ms(timer):
#             return round(1000 * timer.mean, 3)
#
#         result["info"].update({
#             "learner_queue": self.learner_queue_size.stats(),
#             "learner": copy.deepcopy(self.stats),
#             "timing_breakdown": {
#                 "learner_grad_time_ms": timer_to_ms(self.grad_timer),
#                 "learner_load_time_ms": timer_to_ms(self.load_timer),
#                 "learner_load_wait_time_ms": timer_to_ms(self.load_wait_timer),
#                 "learner_dequeue_time_ms": timer_to_ms(self.queue_timer),
#             }
#         })
#         return result
#
# @ray.remote
# class Rollout:
#
#
# class MultiGPULearnerThread(LearnerThread):
#     """Learner that can use multiple GPUs and parallel loading.
#     This class is used for async sampling algorithms.
#     Example workflow: 2 GPUs and 3 multi-GPU tower stacks.
#     -> On each GPU, there are 3 slots for batches, indexed 0, 1, and 2.
#     Workers collect data from env and push it into inqueue:
#     Workers -> (data) -> self.inqueue
#     We also have two queues, indicating, which stacks are loaded and which
#     are not.
#     - idle_tower_stacks = [0, 1, 2]  <- all 3 stacks are free at first.
#     - ready_tower_stacks = []  <- None of the 3 stacks is loaded with data.
#     `ready_tower_stacks` is managed by `ready_tower_stacks_buffer` for
#     possible minibatch-SGD iterations per loaded batch (this avoids a reload
#     from CPU to GPU for each SGD iter).
#     n _MultiGPULoaderThreads: self.inqueue -get()->
#     policy.load_batch_into_buffer() -> ready_stacks = [0 ...]
#     This thread: self.ready_tower_stacks_buffer -get()->
#     policy.learn_on_loaded_batch() -> if SGD-iters done,
#     put stack index back in idle_tower_stacks queue.
#     """
#
#     def __init__(
#             self,
#             local_worker: RolloutWorker,
#             num_gpus: int = 1,
#             lr=None,  # deprecated.
#             train_batch_size: int = 500,
#             num_multi_gpu_tower_stacks: int = 1,
#             num_sgd_iter: int = 1,
#             learner_queue_size: int = 16,
#             learner_queue_timeout: int = 300,
#             num_data_load_threads: int = 16,
#             _fake_gpus: bool = False,
#             # Deprecated arg, use
#             minibatch_buffer_size=None,
#     ):
#         """Initializes a MultiGPULearnerThread instance.
#         Args:
#             local_worker (RolloutWorker): Local RolloutWorker holding
#                 policies this thread will call `load_batch_into_buffer` and
#                 `learn_on_loaded_batch` on.
#             num_gpus (int): Number of GPUs to use for data-parallel SGD.
#             train_batch_size (int): Size of batches (minibatches if
#                 `num_sgd_iter` > 1) to learn on.
#             num_multi_gpu_tower_stacks (int): Number of buffers to parallelly
#                 load data into on one device. Each buffer is of size of
#                 `train_batch_size` and hence increases GPU memory usage
#                 accordingly.
#             num_sgd_iter (int): Number of passes to learn on per train batch
#                 (minibatch if `num_sgd_iter` > 1).
#             learner_queue_size (int): Max size of queue of inbound
#                 train batches to this thread.
#             num_data_load_threads (int): Number of threads to use to load
#                 data into GPU memory in parallel.
#         """
#         # Deprecated: No need to specify as we don't need the actual
#         # minibatch-buffer anyways.
#         if minibatch_buffer_size:
#             deprecation_warning(
#                 old="MultiGPULearnerThread.minibatch_buffer_size",
#                 error=False,
#             )
#         super().__init__(
#             local_worker=local_worker,
#             minibatch_buffer_size=0,
#             num_sgd_iter=num_sgd_iter,
#             learner_queue_size=learner_queue_size,
#             learner_queue_timeout=learner_queue_timeout,
#         )
#         # Delete reference to parent's minibatch_buffer, which is not needed.
#         # Instead, in multi-GPU mode, we pull tower stack indices from the
#         # `self.ready_tower_stacks_buffer` buffer, whose size is exactly
#         # `num_multi_gpu_tower_stacks`.
#         self.minibatch_buffer = None
#
#         self.train_batch_size = train_batch_size
#
#         # TODO: (sven) Allow multi-GPU to work for multi-agent as well.
#         self.policy = self.local_worker.policy_map[DEFAULT_POLICY_ID]
#
#         logger.info("MultiGPULearnerThread devices {}".format(
#             self.policy.devices))
#         assert self.train_batch_size % len(self.policy.devices) == 0
#         assert self.train_batch_size >= len(self.policy.devices), \
#             "batch too small"
#
#         if set(self.local_worker.policy_map.keys()) != {DEFAULT_POLICY_ID}:
#             raise NotImplementedError("Multi-gpu mode for multi-agent")
#
#         self.tower_stack_indices = list(range(num_multi_gpu_tower_stacks))
#
#         # Two queues for tower stacks:
#         # a) Those that are loaded with data ("ready")
#         # b) Those that are ready to be loaded with new data ("idle").
#         self.idle_tower_stacks = queue.Queue()
#         self.ready_tower_stacks = queue.Queue()
#         # In the beginning, all stacks are idle (no loading has taken place
#         # yet).
#         for idx in self.tower_stack_indices:
#             self.idle_tower_stacks.put(idx)
#         # Start n threads that are responsible for loading data into the
#         # different (idle) stacks.
#         for i in range(num_data_load_threads):
#             self.loader_thread = _MultiGPULoaderThread(
#                 self, share_stats=(i == 0))
#             self.loader_thread.start()
#
#         # Create a buffer that holds stack indices that are "ready"
#         # (loaded with data). Those are stacks that we can call
#         # "learn_on_loaded_batch" on.
#         self.ready_tower_stacks_buffer = MinibatchBuffer(
#             self.ready_tower_stacks, num_multi_gpu_tower_stacks,
#             learner_queue_timeout, num_sgd_iter)
#
#     @override(LearnerThread)
#     def step(self) -> None:
#         assert self.loader_thread.is_alive()
#         with self.load_wait_timer:
#             buffer_idx, released = self.ready_tower_stacks_buffer.get()
#
#         with self.grad_timer:
#             fetches = self.policy.learn_on_loaded_batch(
#                 offset=0, buffer_index=buffer_idx)
#             self.weights_updated = True
#             self.stats = {DEFAULT_POLICY_ID: get_learner_stats(fetches)}
#
#         if released:
#             self.idle_tower_stacks.put(buffer_idx)
#
#         self.outqueue.put(
#             (self.policy.get_num_samples_loaded_into_buffer(buffer_idx),
#              self.stats))
#         self.learner_queue_size.push(self.inqueue.qsize())
#
#
# class _MultiGPULoaderThread(threading.Thread):
#     def __init__(self, multi_gpu_learner_thread: MultiGPULearnerThread,
#                  share_stats: bool):
#         threading.Thread.__init__(self)
#         self.multi_gpu_learner_thread = multi_gpu_learner_thread
#         self.daemon = True
#         if share_stats:
#             self.queue_timer = multi_gpu_learner_thread.queue_timer
#             self.load_timer = multi_gpu_learner_thread.load_timer
#         else:
#             self.queue_timer = TimerStat()
#             self.load_timer = TimerStat()
#
#     def run(self) -> None:
#         while True:
#             self._step()
#
#     def _step(self) -> None:
#         s = self.multi_gpu_learner_thread
#         policy = s.policy
#
#         # Get a new batch from the data (inqueue).
#         with self.queue_timer:
#             batch = s.inqueue.get()
#
#         # Get next idle stack for loading.
#         buffer_idx = s.idle_tower_stacks.get()
#
#         # Load the batch into the idle stack.
#         with self.load_timer:
#             policy.load_batch_into_buffer(batch=batch, buffer_index=buffer_idx)
#
#         # Tag just-loaded stack as "ready".
#         s.ready_tower_stacks.put(buffer_idx)
