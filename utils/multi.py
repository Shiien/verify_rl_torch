import logging
import time
import typing
import torch
import torch.multiprocessing as mp
import threading
import os
from utils.environment import Environment
import utils.atari_wrapper as atari_wrappers
from utils import prof

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def create_env(flags):
    return atari_wrappers.wrap_pytorch(
        atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(flags.env),
            clip_rewards=False,
            frame_stack=True,
            scale=False,
        )
    )


def create_buffers(unroll_length, num_buffers, obs_shape, num_actions) -> Buffers:
    T = unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        q=dict(size=(T + 1, num_actions), dtype=torch.float32),
        q_action=dict(size=(T + 1,), dtype=torch.float32),
        action=dict(size=(T + 1,), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def get_batch(
        flags,
        free_queue: mp.SimpleQueue,
        full_queue: mp.SimpleQueue,
        buffers: Buffers,
        initial_agent_state_buffers,
        timings,
        lock=threading.Lock(),
):
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(
        t.to(device=flags.device, non_blocking=True) for t in initial_agent_state
    )
    timings.time("device")
    return batch, initial_agent_state


def act(
        flags,
        actor_index: int,
        free_queue: mp.SimpleQueue,
        full_queue: mp.SimpleQueue,
        model: torch.nn.Module,
        buffers: Buffers,
        initial_agent_state_buffers,
        logging
):
    try:
        logging.info('Actor %i start', actor_index)
        timings = prof.Timings()
        gym_env = create_env(flags)
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        gym_env.seed(seed)
        env = Environment(gym_env)
        env_output = env.initial()
        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state = model(env_output, agent_state)
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            # print(actor_index,'write old rollout end env')
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]

            # print(actor_index,'write old rollout end agent')
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            for t in range(flags.unroll_length):
                timings.reset()
                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)
                timings.time("model")
                env_output = env.step(agent_output["action"])

                timings.time("step")
                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                # print('write env')
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]
                # print('write agent')

                timings.time("write")
            full_queue.put(index)
            if actor_index == 0:
                logging.info("Actor %i: %s", actor_index, timings.summary())
            # logging.info(,actor_index)


    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        print()
        raise e
