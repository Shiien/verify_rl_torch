import gym
import cv2
import numpy as np
from collections import deque
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.wrappers.time_limit import TimeLimit

cv2.ocl.setUseOpenCL(False)


def make_atari_by_id(env_id, max_episode_steps=None):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if max_episode_steps is not None and max_episode_steps > 0:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def make_atari(env, max_episode_steps=None):
    assert 'NoFrameskip' in env.spec.id
    # print('set time limit:', max_episode_steps)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if max_episode_steps is not None and max_episode_steps > 0:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def make_atari_ram(env, max_episode_steps, scale=True, episode_life=True):
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if max_episode_steps > 0:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    if scale:
        env = ScaledFloatFrame(env)
    if episode_life:
        env = EpisodicLifeEnv(env)
    return env


def make_minatar(env, max_episode_steps, scale=False):
    if max_episode_steps > 0:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    if scale:
        env = ScaledFloatFrame(env)
    return env


def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    # Configure environment for DeepMind-style Atari.
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    """

    def __init__(self, env, noop_max=30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        # Do no-op action for a number of steps in [1, noop_max]
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class MaxAndSkipEnv(gym.Wrapper):
    # Return only every skip-th frame
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        # Observation buffer to store most recent raw observations
        # for max pooling across time steps
        self.obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self.skip = skip

    def step(self, action):
        # Repeat action, sum reward, and max over last observations
        total_reward = 0.0
        done = None
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            if i == self.skip - 2:
                self.obs_buffer[0] = obs
            elif i == self.skip - 1:
                self.obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self.obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FireResetEnv(gym.Wrapper):
    # Take action on reset for environments that are fixed until firing
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # Check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class WarpFrame(gym.ObservationWrapper):
    """
    Warp frames to 84x84 as done in the Nature paper and later work.
    If the environment uses dictionary observations, `dict_spacekey` can be
    specified which indicates which observation should be warped.
    """

    def __init__(self, env, width=84, height=84, grayscale=True, dict_spacekey=None):
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.key = dict_spacekey
        if self.grayscale:
            num_colors = 1
        else:
            num_colors = 3
        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, num_colors),
            dtype=np.uint8
        )
        if self.key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self.key]
            self.observation_space.spaces[self.key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        frame = obs if self.key is None else obs[self.key]
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        if self.key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self.key] = frame
        return obs


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # Careful! This undoes the memory optimization,
        # use with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class ClipRewardEnv(gym.RewardWrapper):
    # Bin reward to {+1, 0, -1} by its sign.
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        return np.sign(reward)


class FrameStack(gym.Wrapper):
    """
    Stack k last frames.
    Returns lazy array, which is much more memory efficient.
    """

    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=((shp[0] * k,) + shp[1:]),
                                                dtype=env.observation_space.dtype)
        # shape = (shp[:-1]+(shp[-1] * k,))

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    """
    This object ensures that common frames between the observations are only stored once.
    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
    buffers.
    This object should only be converted to numpy array before being passed to the model.
    You'd not believe how complex the previous solution was.
    """

    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[..., i]


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class ReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        if done:
            info['episodic_return'] = self.total_rewards
            self.total_rewards = 0
        else:
            info['episodic_return'] = None
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))


def wrap_pytorch(env):
    return ImageToPyTorch(env)
