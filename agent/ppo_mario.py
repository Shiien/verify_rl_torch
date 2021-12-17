import threading
import time
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from six.moves.queue import Queue, Empty
import gym
from torch.utils.tensorboard import SummaryWriter
import os
import copy
import numpy as np


@ray.remote(num_gpus=0)
class Rollout(threading.Thread):
    def __init__(self, model_fn, env_fn, config):
        super().__init__()
        self.env: gym.Env = env_fn()
        self.model: torch.nn.Module = model_fn()
        self.running = False
        self.queue = Queue(maxsize=50)
        self.weight_queue = Queue(maxsize=1)
        self.config = config

    def get_temp_dict(self, t, x):
        batch = {
            'state': torch.zeros([t, x]),
            'action': torch.zeros([t]).long(),
            'reward': torch.zeros([t]),
            'done': torch.zeros([t, ]).bool(),
            'logp': torch.zeros([t]).float(),
            'value': torch.zeros([t, ]),
            'G': torch.zeros([t, ]),
            'adv': torch.zeros([t, ]),
            'return': torch.zeros([t, ]),
            'gamma': torch.zeros([t, ]),
            'init_h': self.model.get_hidden()
        }
        return batch

    def run(self):
        self.running = True
        observation_shape = self.env.reset().shape[-1]
        temp_batch = self.get_temp_dict(self.config.num_unroll_steps, observation_shape)
        Tidx = 0
        while self.running:
            idx = 0
            batch = self.get_temp_dict(self.config.max_length, observation_shape)
            h = self.model.get_hidden()
            hs = []
            s = self.env.reset()
            re = 0
            done = False
            while not done:
                batch['s'][idx] = torch.Tensor(s).float()
                batch['valid_action'][idx] = torch.from_numpy(mask_to_vector(self.env.legal_actions(),
                                                                             self.config.action_space)).float()
                hs.append(h)
                with torch.no_grad():
                    (a, logp), v, h = self.model.act(
                        batch['s'][idx].unsqueeze(0), batch['valid_action'][idx].unsqueeze(0), h=h)
                batch['a'][idx] = a.item()
                s, reward, done, _ = self.env.step(a.item())
                re += reward
                batch['r'][idx] = torch.FloatTensor([reward])
                batch['d'][idx] = torch.BoolTensor([done])
                batch['v'][idx] = v.view(1)
                batch['logp'][idx] = logp.view(-1)
                if done:
                    batch['return'][idx] = torch.FloatTensor([re])
                    try:
                        weight = self.weight_queue.get_nowait()
                        self.model.load_state_dict(weight)
                    except Empty:
                        pass
                idx += 1
            lastgaelam = 0
            for i in range(idx - 1, -1, -1):
                if i == idx - 1:
                    next_v = 0
                    next_e = 0
                    nextnonterminal = False
                else:
                    next_v = batch['v'][i + 1]
                    next_e = batch['r'][i + 1]
                    nextnonterminal = True
                delta = batch['r'][i] + self.config.discount * next_v * nextnonterminal - batch['v'][i]
                batch['adv'][
                    i] = lastgaelam = delta + self.config.discount * self.config.lambda_ * nextnonterminal * lastgaelam
                batch['re'][i] = batch['r'][i] + self.config.discount * next_e
            for i in range(idx):
                for k in batch.keys():
                    if k != 'init_h':
                        temp_batch[k][Tidx] = copy.deepcopy(batch[k][i])
                Tidx += 1
                if Tidx == self.config.num_unroll_steps:
                    self.queue.put(temp_batch, timeout=600)
                    temp_batch = self.get_temp_dict(self.config.num_unroll_steps, observation_shape)
                    for k in batch.keys():
                        if k != 'init_h':
                            temp_batch[k][0] = copy.deepcopy(batch[k][i])
                    temp_batch['init_h'] = hs[i]
                    Tidx = 1

    def get_data(self):
        if not self.is_alive():
            raise RuntimeError("thread has died")
        return self.queue.get(timeout=600)

    def set_parameter(self, parameter):
        if self.weight_queue.empty():
            self.weight_queue.put(parameter)
        else:
            self.weight_queue.get()
            self.weight_queue.put(parameter)

    def set_parameter_directly(self, parameter):
        self.model.load_state_dict(parameter)

    def set_running(self, running=False):
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
            data = {k: v.to(device=self.device) if isinstance(v, torch.Tensor) or k == 'graph' else tuple(
                i.to(device=self.device) for i in v) for k, v in data.items()}
            self.queue.put(data)


class PPOLearner():
    def __init__(self, model_fn, env_fn, config):
        self.env = env_fn()
        self.model: nn.Module = model_fn()
        self.config = config

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config.lr_init,
                                      weight_decay=self.config.weight_decay)
        ray.init(num_cpus=self.config.num_workers, num_gpus=self.config.max_num_gpus)

        def lr_lambda(epoch):
            return 1 - min(epoch * config.num_unroll_steps * config.batch_size,
                           config.training_steps) / config.training_steps

        self.tune = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)
        self.running = False
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.train_on_gpu else "cpu")
        self.model.to(self.device)
        self.queue = Queue(maxsize=config.batch_size * 5)
        self.ok_queue = Queue(maxsize=5)
        self.lock = threading.Lock()
        self.action = 0
        self.batch_size = config.batch_size
        actors = []
        for _ in range(config.num_workers):
            actor = Rollout.remote(model_fn=model_fn, env_fn=env_fn, config=self.config)
            actors.append(actor)
        self.actors = actors
        self.logger = None
        if self.config.save_log:
            self.logger = SummaryWriter(
                log_dir=self.config.results_path,
                flush_secs=5)
        self.worker = [Batcher(actor, self.device, self.queue) for actor in actors]

    def init_learn(self):
        for work in self.worker:
            work.setDaemon(True)
        weight = ray.put(copy.deepcopy(self.model.get_weights()))  # .remote()
        ray.get([actor.set_parameter_directly.remote(weight) for actor in self.actors])
        for actor in self.actors:
            actor.start.remote()
        for work in self.worker:
            work.start()
        self.collect_worker = threading.Thread(target=self.get_batch,
                                               args=(self.queue, self.ok_queue, self.device, self.batch_size),
                                               daemon=True)
        self.collect_worker.start()

        print('init ok')
        self.run()

        ray.shutdown()

    @staticmethod
    def get_batch(queue, ok_queue, device, batch_size):
        while True:
            datas = []
            for i in range(batch_size):
                datas.append(queue.get(timeout=600))
            new_datas = {}
            for key in datas[0].keys():
                if key != 'init_h' and key != 'graph':
                    new_datas[key] = torch.stack([i[key] for i in datas]).transpose(1, 0)
                if key == 'graph':
                    new_datas[key] = [i[key] for i in datas]
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
        config = self.config
        try:
            while True:
                get_time = time.time()
                datas = self.ok_queue.get()
                get_time = time.time() - get_time
                learn_time = time.time()
                stat = self.learn_from_batch(datas)
                learn_time = time.time() - learn_time
                self.step += config.batch_size * config.num_unroll_steps
                print(self.step // (config.batch_size * config.num_unroll_steps), stat)
                if self.logger is not None:
                    for key, value in stat.items():
                        if key != 'p_hist' and not np.isnan(value):
                            self.logger.add_scalar(f'train/{key}', value, self.step)
                        if key == 'p_hist':
                            self.logger.add_histogram(f'train/{key}', value, self.step)
                    self.logger.add_scalar('time/get_time', get_time, self.step)
                    self.logger.add_scalar('time/learn_time', learn_time, self.step)
                if self.step >= config.training_steps:
                    ray.get([actor.set_running.remote(False) for actor in self.actors])
                    ray.get([actor.join.remote(False) for actor in self.actors])
                    # for actor in self.actors:
                    #     actor.join.remote()
                    print('actor stop ok')
                    break
                tstep += config.batch_size * config.num_unroll_steps
                if tstep > 1e5:
                    f = open(os.path.join(self.config.results_path, 'model_last1.pth'), 'wb')
                    torch.save(self.model.state_dict(), f)
                    tstep = 0
                if (self.step // (config.batch_size * config.num_unroll_steps)) % self.config.checkpoint_interval == 0:
                    weight = ray.put(copy.deepcopy(self.model.get_weights()))
                    [actor.set_parameter.remote(weight) for actor in self.actors]
        except KeyboardInterrupt:
            ray.get([actor.set_running.remote(False) for actor in self.actors])
            for actor in self.actors:
                actor.join()
            print('actor stop ok')

    def learn_from_batch(self, batch):

        # learner_outputs = self.model.forward(batch)
        adv = batch['adv'][:-1]  # + self.config['discount'] * learner_outputs['v'][1:] - learner_outputs['v'][:-1]
        (v, logp) = self.model.get_learn_output(batch)
        logp = logp[:-1]
        v = v[:-1]
        log_ratio = (logp - batch['logp'][:-1])  # .sum(-1)
        ratio = log_ratio.exp()  # .clamp(1e-3, 1000)
        ratio = ratio.reshape(-1)
        adv = adv.reshape(-1)
        logp = logp.reshape(-1)
        adv_targ = adv
        adv_targ = (adv_targ - adv_targ.mean()) / (adv_targ.std() + 1e-8)
        adv_targ = adv_targ.detach()
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_param,
                            1.0 + self.config.clip_param) * adv_targ
        actor_loss = -torch.min(surr1, surr2).mean() + self.config.cofentropy * (-logp.mean())
        v = v.reshape(-1)
        re = batch['re'][:-1].reshape(-1)
        critic_loss = F.smooth_l1_loss(v, re)
        total_loss = actor_loss + self.config.v_scaling * critic_loss
        episode_returns = batch["return"][batch["d"]]
        try:
            tmp_dict = {
                'return': episode_returns.mean().item(),
                'return_max': episode_returns.max().item(),
                'return_min': episode_returns.min().item(),
            }
        except:
            tmp_dict = {}
        self.optim.zero_grad()
        total_loss.backward()
        torch.nn.NLLLoss()
        grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optim.step()
        # self.tune.step()
        ratio_count = ratio.le(1.0 + self.config.clip_param).bool() & ratio.ge(1.0 - self.config.clip_param).bool()
        ratio_count = ratio_count.float().mean().item()
        return {
            'total_loss': total_loss.item(),
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'adv': adv.mean().item(),
            'v': v.mean().item(),
            'entropy': (-logp.mean()).item(),
            'p': logp.exp().mean().item(),
            'p_std': logp.exp().std().item(),
            # 'p_max': logp.exp().max().item(),
            # 'p_min': logp.exp().min().item(),
            'ratio_std': ratio.std().item(),
            'ratio_c': ratio_count,
            'p_hist': logp.exp().detach().cpu(),
            # 'lr': self.optim.state.get('lr', 0),
            'grad': grad.norm().item(),
            **tmp_dict
        }

    def set_parameter(self, parameter):
        with self.lock:
            self.action = parameter

    def set_running(self, running):
        self.running = running
