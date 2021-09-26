import threading
import time

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from six.moves.queue import Queue
import utils.some_loss as some_loss
import utils.some_trace as some_trace
import gym
from functools import partial
from utils.atari_wrapper import make_atari_by_id, wrap_deepmind, wrap_pytorch
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from model.atari_impala import AtariPolicy, MiniAtariPolicy, MLPPolicy
import numpy as np
from collections import namedtuple


# import matplotlib
# print(matplotlib.use('TkAgg'))
# assert 0
# def create_env():
#     return wrap_pytorch(
#         wrap_deepmind(
#             make_atari_by_id(env_id='PongNoFrameskip-v4'),
#             clip_rewards=False,
#             frame_stack=True,
#             scale=True,
#         )
#     )

def create_env():
    return gym.make('LunarLander-v2')


class Miniatari:
    def __init__(self, game_name='breakout'):
        'asterix,breakout,freeway,seaquest,space_invaders'
        from minatar import Environment
        self.env = Environment(game_name)
        self.action_space = namedtuple('action_space', 'n')
        self.action_space.n = self.env.num_actions()

    def reset(self):
        self.env.reset()
        return torch.tensor(self.env.state()).permute(2, 0, 1).numpy()

    def step(self, action):
        r, d = self.env.act(action)
        s = torch.tensor(self.env.state()).permute(2, 0, 1).numpy()
        return s, r, d, None


# def create_env():
#     return Miniatari()


BATCHSIZE = 256
TOTAL_STEP = int(10 * 1e6)
TT = 40
env_fn = create_env
env = env_fn()

model_fn = partial(MLPPolicy, observation_shape=(8), mlp_dims=(8, 64, 64), num_actions=env.action_space.n,
                   use_lstm=True)


# print(env.reset().shape)
# assert 0


@ray.remote
class Rollout(threading.Thread):
    def __init__(self, model_fn, env_fn, observation_shape, T):
        super().__init__()
        self.env: gym.Env = env_fn()
        self.model: torch.nn.Module = model_fn()
        self.T = T
        self.running = False
        self.queue = Queue(maxsize=50)
        self.weight_queue = Queue(maxsize=1)
        self.lock = threading.Lock()
        self.observation_shape = observation_shape

    def run(self):
        self.running = True
        s = self.env.reset()
        re = 0
        while self.running:
            batch = {
                's': torch.zeros([self.T, *self.observation_shape]).float(),
                'a': torch.zeros([self.T]).long(),
                'r': torch.zeros([self.T]),
                'd': torch.zeros([self.T, ]).bool(),
                'logp': torch.zeros([self.T, self.env.action_space.n]),
                'v': torch.zeros([self.T, ]),
                'return': torch.zeros([self.T, ]),
                'init_h': self.model.initial_state(1)
            }
            h = self.model.initial_state(1)
            for idx in range(self.T):
                batch['s'][idx] = torch.Tensor(s)
                if idx == 0:
                    batch['init_h'] = h
                with self.lock:
                    with torch.no_grad():
                        output = self.model.forward(
                            {'s': torch.Tensor(s).unsqueeze(0).unsqueeze(0), 'init_h': h,
                             'd': torch.BoolTensor([[[False]]])})
                a, v, logp, h = output['a'], output['v'], output['logp'], output['init_h']
                batch['a'][idx] = torch.LongTensor([a.item()])
                s, r, d, info = self.env.step(a.item())
                re += r
                batch['r'][idx] = torch.FloatTensor([r])/200.0
                batch['d'][idx] = torch.BoolTensor([d])
                batch['v'][idx] = v.view(1)  # .clone()
                batch['logp'][idx] = logp.view(-1)  # .clone()
                if d:
                    batch['return'][idx] = torch.FloatTensor([re])
                    re = 0
                    s = self.env.reset()
                    try:
                        with self.lock:
                            weight = self.weight_queue.get_nowait()
                            self.model.load_state_dict(weight)
                    except:
                        pass
            self.queue.put(batch, timeout=600)

    def get_data(self):
        if not self.is_alive():
            raise RuntimeError("thread has died")
        return self.queue.get(timeout=600)

    def set_parameter(self, parameter):
        with self.lock:
            if self.weight_queue.empty():
                self.weight_queue.put(parameter)
            else:
                self.weight_queue.get()
                self.weight_queue.put(parameter)

    def set_parameter_directly(self, parameter):
        with self.lock:
            self.model.load_state_dict(parameter)

    def set_running(self, running=False):
        self.running = running


ray.init()


class Batcher(threading.Thread):
    def __init__(self, actor, device, queue):
        super(Batcher, self).__init__()
        self.actors = actor
        self.queue = queue
        self.device = device

    def run(self) -> None:
        while True:
            data: dict = ray.get(self.actors.get_data.remote())
            data = {k: v.to(device=self.device, non_blocking=True) if isinstance(v, torch.Tensor) else tuple(
                i.to(device=self.device, non_blocking=True) for i in v) for k, v in data.items()}
            self.queue.put(data)


class Learner(threading.Thread):
    def __init__(self, model_fn, env_fn, device, batch_size, config, actors):
        super().__init__()
        self.env = env_fn()
        self.model: nn.Module = model_fn()
        # self.optim = torch.optim.RMSprop(self.model.parameters(), lr=0.0006, momentum=0, eps=0.01)

        self.optim = torch.optim.SGD(self.model.parameters(), lr=1e-3)

        def lr_lambda(epoch):
            return 1 - min(epoch * TT * BATCHSIZE, TOTAL_STEP) / TOTAL_STEP

        # self.tune = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)
        self.running = False
        self.device = device
        self.model.to(self.device)
        self.queue = Queue(maxsize=BATCHSIZE * 5)
        self.ok_queue = Queue(maxsize=5)
        self.lock = threading.Lock()
        self.action = 0
        self.batch_size = batch_size
        self.actors = actors
        self.config = config
        self.logger = None
        if self.config.get('save_log', False):
            assert 'log_path' in self.config.keys(), 'need log_path'
            self.logger = SummaryWriter(
                log_dir=os.path.join(self.config['log_path'], datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')),
                flush_secs=5)
        self.worker = [Batcher(actor, self.device, self.queue) for actor in actors]

    def init(self):
        for work in self.worker:
            work.setDaemon(True)
        weight = ray.put(self.model.get_parameter())  # .remote()
        ray.get([actor.set_parameter_directly.remote(weight) for actor in actors])
        for actor in self.actors:
            actor.start.remote()
        for work in self.worker:
            work.start()
        self.collect_worker = threading.Thread(target=self.get_batch,
                                               args=(self.queue, self.ok_queue, self.device, self.batch_size),
                                               daemon=True)
        self.collect_worker.start()
        self.start()
        print('init ok')

    @staticmethod
    def get_batch(queue, ok_queue, device, batch_size):
        while True:
            datas = []
            for i in range(batch_size):
                datas.append(queue.get(timeout=600))
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
                    t.to(device=device, non_blocking=True) for t in init_h
                )
            ok_queue.put(new_datas)

    def run(self):
        self.step = 0
        tstep = 0
        try:
            while True:
                get_time = time.time()
                datas = self.ok_queue.get()
                get_time = time.time() - get_time
                learn_time = time.time()
                stat = self.learn_from_batch(datas)
                learn_time = time.time() - learn_time
                self.step += TT * BATCHSIZE
                print(self.step, stat)
                if self.logger is not None:
                    for key, value in stat.items():
                        if not np.isnan(value):
                            self.logger.add_scalar(f'train/{key}', value, self.step)
                    self.logger.add_scalar('time/get_time', get_time, self.step)
                    self.logger.add_scalar('time/learn_time', learn_time, self.step)
                if self.step >= TOTAL_STEP:
                    ray.get([actor.set_running.remote(False) for actor in self.actors])
                    ray.get([actor.join.remote(False) for actor in self.actors])
                    # for actor in self.actors:
                    #     actor.join.remote()
                    print('actor stop ok')
                    break
                tstep += TT * BATCHSIZE
                if tstep > 1e6:
                    f = open(f'save_model/model_last.pth', 'wb')
                    torch.save(self.model.state_dict(), f)
                    tstep = 0
                if (self.step // (TT * BATCHSIZE)) % 32 == 0:
                    weight = ray.put(self.model.get_parameter())
                    [actor.set_parameter.remote(weight) for actor in actors]
        except KeyboardInterrupt:
            ray.get([actor.set_running.remote(False) for actor in self.actors])
            for actor in self.actors:
                actor.join()
            print('actor stop ok')

    def learn_from_batch(self, batch):
        learner_outputs = self.model.forward(batch)
        bootstrap_value = learner_outputs["v"][-1]
        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items() if key != 'init_h'}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        rewards = batch["r"]  # .clip(-1, 1)
        discounts = (~batch["d"]).float() * self.config['discount']

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
        # pg_loss = some_loss.compute_ppo_loss(learner_outputs['logp'],
        #                                      batch['logp'], batch['a'], vtrace_returns.pg_advantages.detach())
        pg_loss = some_loss.compute_policy_gradient_loss(
            learner_outputs["logp"],
            batch["a"],
            vtrace_returns.pg_advantages.detach(),
        )
        baseline_loss = self.config['v_scaling'] * some_loss.compute_baseline_loss(
            vtrace_returns.vs.detach() - learner_outputs["v"]
        )
        entropy_loss = self.config['entropy_scaling'] * some_loss.compute_entropy_loss(
            learner_outputs["logp"]
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch["return"][batch["d"]]

        self.optim.zero_grad()
        total_loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optim.step()
        # self.tune.step()
        return {
            'pg_loss': pg_loss.item(),
            'v_loss': baseline_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'return': episode_returns.mean().item(),
            'v_mean': vtrace_returns.vs.mean().item(),
            'v_max': vtrace_returns.vs.max().item(),
            'v_min': vtrace_returns.vs.min().item(),
            'rho': vtrace_returns.log_rhos.exp().mean().item(),
            'rho_min': vtrace_returns.log_rhos.exp().min().item(),
            'rho_max': vtrace_returns.log_rhos.exp().max().item(),
            'r_mean': rewards.mean().item(),
            'grad': grad.norm().item()
        }

    def set_parameter(self, parameter):
        with self.lock:
            self.action = parameter

    def set_running(self, running):
        self.running = running


actors = []
config = {
    'entropy_scaling': 0.01,
    'v_scaling': 0.5,
    'discount': 0.99,
    'log_path': './logs',
    'save_log': True,
}
for _ in range(16):
    actor = Rollout.remote(model_fn=model_fn, env_fn=env_fn, observation_shape=(8,), T=TT + 1)
    actors.append(actor)
learner = Learner(model_fn=model_fn, env_fn=env_fn, device='cuda', batch_size=BATCHSIZE, config=config, actors=actors)
learner.init()
learner.join()
ray.shutdown()
