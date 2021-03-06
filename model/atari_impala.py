import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.layer import layer_init
from torch.distributions import Bernoulli


class AtariNet(nn.Module):
    def __init__(self, observation_shape, num_actions, use_lstm=False):
        super(AtariNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        # Feature extraction.
        self.conv1 = nn.Conv2d(
            in_channels=self.observation_shape[0],
            out_channels=32,
            kernel_size=8,
            stride=4,
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc = nn.Linear(3136, 512)

        core_output_size = self.fc.out_features + num_actions + 1

        self.use_lstm = use_lstm
        if use_lstm:
            self.core = nn.LSTM(core_output_size, core_output_size, 2)

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state=()):
        x = inputs["frame"]  # [T, B, C, H, W].
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        one_hot_last_action = F.one_hot(
            inputs["last_action"].view(T * B), self.num_actions
        ).float()
        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )


class AtariCasaNet(nn.Module):
    def __init__(self, observation_shape, num_actions, use_lstm=False):
        super(AtariCasaNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        # Feature extraction.
        self.conv1 = nn.Conv2d(
            in_channels=self.observation_shape[0],
            out_channels=32,
            kernel_size=8,
            stride=4,
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc = nn.Linear(3136, 512)

        core_output_size = self.fc.out_features + num_actions + 1

        self.use_lstm = use_lstm
        if use_lstm:
            self.core = nn.LSTM(core_output_size, core_output_size, 2)

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state=()):
        x = inputs["frame"]  # [T, B, C, H, W].
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        one_hot_last_action = F.one_hot(
            inputs["last_action"].view(T * B), self.num_actions
        ).float()
        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        A = self.policy(core_output)
        policy_prob = F.softmax(A, dim=1)
        policy_logits = A
        # A_bar= A
        A_bar = (A * policy_prob.detach()).sum(dim=1, keepdims=True)
        baseline = self.baseline(core_output)
        q = A - A_bar + baseline.detach()
        if self.training:
            dist = torch.distributions.Categorical(policy_prob)
            action = dist.sample()
        else:
            action = torch.argmax(policy_prob, dim=1)
        q_action = q.gather(-1, action.unsqueeze(-1))
        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        q = q.view(T, B, self.num_actions)
        q_action = q_action.view(T, B)
        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action, q=q, q_action=q_action),
            core_state,
        )


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class AtariPolicy(torch.nn.Module):
    def __init__(self, observation_shape, num_actions, use_lstm=False):
        super().__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        # Feature extraction.
        self.conv1 = init_(nn.Conv2d(
            in_channels=self.observation_shape[0],
            out_channels=32,
            kernel_size=8,
            stride=4,
        ))
        self.conv2 = init_(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = init_(nn.Conv2d(64, 64, kernel_size=3, stride=1))

        self.fc = init_(nn.Linear(3136, 512))

        core_output_size = self.fc.out_features  # + num_actions + 1

        self.use_lstm = use_lstm
        if use_lstm:
            self.core = nn.LSTM(core_output_size, core_output_size, 2)

        self.policy = init_(nn.Linear(core_output_size, self.num_actions))
        self.baseline = init_(nn.Linear(core_output_size, 1))

    def initial_state(self, batch_size):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs):
        x = inputs["s"]  # [T, B, C, H, W].
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))
        core_input = x
        core_state = inputs['init_h']
        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["d"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            dist = torch.distributions.Categorical(F.softmax(policy_logits, dim=1))
            action = dist.sample()
        else:
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return {'v': baseline, 'a': action, 'logp': policy_logits, 'init_h': core_state}

    def get_parameter(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}


class Conv2d_MinAtar(nn.Module):
    '''
    2D convolution neural network for MinAtar games
    '''

    def __init__(self, in_channels, feature_dim=128):
        super().__init__()
        self.conv1 = layer_init(nn.Conv2d(in_channels, 16, kernel_size=3, stride=1))

        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        linear_input_size = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc2 = layer_init(nn.Linear(linear_input_size, feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = y.view(y.shape[0], -1)
        y = F.relu(self.fc2(y))
        y_f = y.sigmoid(dim=-1)
        y_f = Bernoulli(y_f).sample()
        return y  + y_f - y.detach()


class MiniAtariPolicy(nn.Module):
    def __init__(self, observation_shape, num_actions, trainning=True, use_lstm=False):
        super().__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.training = trainning
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        # Feature extraction.
        self.conv = Conv2d_MinAtar(self.observation_shape[0])

        # self.fc = init_(nn.Linear(3136, 512))

        core_output_size = 128  # self.fc.out_features  # + num_actions + 1

        self.use_lstm = use_lstm
        if use_lstm:
            self.core = nn.LSTM(core_output_size, core_output_size, 2)

        self.policy = init_(nn.Linear(core_output_size, self.num_actions))
        self.baseline = init_(nn.Linear(core_output_size, 1))

    def initial_state(self, batch_size):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs):
        x = inputs["s"]  # [T, B, C, H, W].
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float()
        x = self.conv(x)
        # x = x.view(T * B, -1)
        # x = F.relu(self.fc(x))
        core_input = x
        core_state = inputs['init_h']
        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["d"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            dist = torch.distributions.Categorical(F.softmax(policy_logits, dim=1))
            action = dist.sample()
        else:
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return {'v': baseline, 'a': action, 'logp': policy_logits, 'init_h': core_state}

    def get_parameter(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}


class MLPPolicy(nn.Module):
    def __init__(self, observation_shape,mlp_dims, num_actions, use_lstm=False):
        super(MLPPolicy, self).__init__()
        self.observation_shape=observation_shape
        self.mlp = []
        for i in range(len(mlp_dims) - 1):
            self.mlp.append(layer_init(nn.Linear(mlp_dims[i], mlp_dims[i + 1])))
            if i != len(mlp_dims) - 2:
                self.mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*self.mlp)
        self.num_actions = num_actions
        self.use_lstm = use_lstm
        if use_lstm:
            self.core = nn.LSTM(mlp_dims[-1], mlp_dims[-1], 2)
        self.policy = layer_init(nn.Linear(mlp_dims[-1], self.num_actions))
        self.baseline = layer_init(nn.Linear(mlp_dims[-1], 1))

    def initial_state(self, batch_size):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def get_parameter(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def forward(self, inputs):
        x = inputs["s"]  # [T, B, C, H, W].
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = self.mlp(x)
        core_input = x
        if self.use_lstm:
            core_state = inputs['init_h']
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["d"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            dist = torch.distributions.Categorical(F.softmax(policy_logits, dim=1))
            action = dist.sample()
        else:
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return {'v': baseline, 'a': action, 'logp': policy_logits, 'init_h': core_state}


if __name__ == '__main__':
    inputs = {
        'frame': torch.zeros([80, 2, 4, 84, 84]).to('cuda'),
        'done': torch.zeros([80, 2, 1]).bool().to('cuda'),
        'reward': torch.zeros([80, 2, 1]).to('cuda'),
        'last_action': torch.zeros([80, 2, 1]).long().to('cuda')
    }
    import time

    m1 = AtariNet([4, 84, 84], 6, False).to('cuda')
    m2 = AtariCasaNet([4, 84, 84], 6, False).to('cuda')

    t = time.time()
    for _ in range(200):
        m1.forward(inputs)

    print(time.time() - t)
    t = time.time()
    for _ in range(200):
        m2.forward(inputs)

    print(time.time() - t)
