import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
from model import Q_Actor, ParamNet
from memory import ReplayBuffer
import os


class PDQNAgent:
    def __init__(self, state_space, action_space, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=5000,
                 batch_size=32, gamma=0.90, replay_memory_size=1e5, actor_lr=1e-3, param_net_lr=1e-4,
                 actor_kwargs={}, param_net_kwargs={}, device="cuda" if torch.cuda.is_available() else "cpu",
                 seed=None, loss_function=F.smooth_l1_loss):
        if param_net_kwargs is None:
            param_net_kwargs = {}
        self.action_space = action_space
        self.state_space = state_space
        self.device = torch.device(device)
        self.seed = seed
        random.seed(self.seed)
        self.np_random = np.random.RandomState(seed=seed)
        self.num_actions = action_space.spaces[0].n
        self.actions_count = 0
        self.action_parameter_sizes = np.array(
            [self.action_space.spaces[i].shape[0] for i in range(1, 1 + self.num_actions)])
        self.action_parameter_size = self.action_parameter_sizes.sum()
        self.action_parameter_max_numpy = np.concatenate(
            [self.action_space.spaces[i].high for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_min_numpy = np.concatenate(
            [self.action_space.spaces[i].low for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)

        self.action_max = torch.from_numpy(np.ones((self.num_actions,))).float().to(device)
        self.action_min = -self.action_max.detach()
        self.action_range = (self.action_max - self.action_min).detach()
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(device)

        self.epsilon = 0.1
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.param_net_lr = param_net_lr
        self.actor_net = Q_Actor(self.state_space.n, self.num_actions, self.action_parameter_size,
                                 **actor_kwargs).to(self.device)
        self.actor_target = Q_Actor(self.state_space.n, self.num_actions, self.action_parameter_size,
                                    **actor_kwargs).to(self.device)
        self.actor_target.load_state_dict(self.actor_net.state_dict())
        self.actor_target.eval()  # 不启用 BatchNormalization 和 Dropout

        self.param_net = ParamNet(self.state_space.n, self.num_actions, self.action_parameter_size,
                                  **param_net_kwargs).to(self.device)
        self.param_net_target = ParamNet(self.state_space.n, self.num_actions, self.action_parameter_size,
                                         **param_net_kwargs).to(self.device)
        self.param_net_target.load_state_dict(self.param_net.state_dict())
        self.param_net_target.eval()

        self.loss_func = loss_function
        self.actor_optimiser = optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)
        self.param_net_optimiser = optim.Adam(self.param_net.parameters(), lr=self.param_net_lr)

        self.memory = ReplayBuffer(capacity=replay_memory_size)
        self.clip_grad = 10
        self.update_count = 0

    def __str__(self):
        desc = "P-DQN Agent\n"
        desc += "Actor Network {}\n".format(self.actor_net) + \
                "Param Network {}\n".format(self.param_net) + \
                "Gamma:{}\n".format(self.gamma) + \
                "Batch Size {}\n".format(self.batch_size) + \
                "Seed{}\n".format(self.seed)

        return desc

    def choose_action(self, state, train=True):
        if train:
            self.actions_count += 1
            with torch.no_grad():
                state = torch.tensor(state, device=self.device)
                all_action_parameters = self.param_net.forward(state)

                if random.random() < self.epsilon:
                    # print("random action")
                    action = self.np_random.choice(self.num_actions)
                    all_action_parameters = torch.from_numpy(
                        np.random.uniform(self.action_parameter_min_numpy, self.action_parameter_max_numpy))
                else:
                    Q_value = self.actor_net.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                    Q_value = Q_value.detach().data.cpu().numpy()
                    action = np.argmax(Q_value)

                all_action_parameters = all_action_parameters.cpu().data.numpy()
                offset = np.array([self.action_parameter_sizes[i] for i in range(action)], dtype=int).sum()
                action_parameters = all_action_parameters[offset:offset + self.action_parameter_sizes[action]]
        else:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device)
                all_action_parameters = self.param_net.forward(state)
                Q_value = self.actor_net.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                Q_value = Q_value.detach().data.numpy()
                action = Q_value.max(1)[1].item()
                all_action_parameters = all_action_parameters.cpu().data.numpy()
                offset = np.array([self.action_parameter_sizes[i] for i in range(action)], dtype=int).sum()
                action_parameters = all_action_parameters[offset:offset + self.action_parameter_sizes[action]]

        return action, action_parameters, all_action_parameters

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.from_numpy(states).to(self.device)
        actions_combined = torch.from_numpy(actions).to(self.device)
        actions = actions_combined[:, 0].long()
        action_parameters = actions_combined[:, 1:]
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)

        # -----------------------optimize Q actor------------------------
        with torch.no_grad():
            next_action_parameters = self.param_net_target.forward(next_states)
            q_value_next = self.actor_target(next_states, next_action_parameters)
            q_value_max_next = torch.max(q_value_next, 1, keepdim=True)[0].squeeze()

            target = rewards + (torch.logical_not(dones).float()) * self.gamma * q_value_max_next

        states = torch.tensor(states).float()
        action_parameters = torch.tensor(action_parameters).float()
        q_values = self.actor_net(states, action_parameters)
        y_predicted = q_values.gather(1, actions.view(-1, 1)).squeeze()
        y_expected = target
        loss_actor = self.loss_func(y_predicted, y_expected)

        self.actor_optimiser.zero_grad()
        loss_actor.backward()
        for param in self.actor_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.actor_optimiser.step()

        # ------------------------optimize param net------------------------------
        with torch.no_grad():
            action_params = self.param_net(states)
        action_params.requires_grad = True
        q_val = self.actor_net(states, action_params)
        param_loss = torch.mean(torch.sum(q_val, 1))
        self.actor_net.zero_grad()
        param_loss.backward()
        # self.param_net_optimiser.step()
        # todo parameter_loss design
        from copy import deepcopy
        delta_a = deepcopy(action_params.grad.data)
        # step 2
        action_params = self.param_net(Variable(states))
        delta_a[:] = self._invert_gradients(delta_a, action_params, grad_type="action_parameters", inplace=True)

        out = -torch.mul(delta_a, action_params)
        self.param_net.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.param_net.parameters(), self.clip_grad)
        self.param_net_optimiser.step()
        self.update_count += 1
        if self.update_count % 100 == 0:
            self.actor_target.load_state_dict(self.actor_net.state_dict())
            self.param_net_target.load_state_dict(self.param_net.state_dict())

    def save(self, name):
        weight_path = 'PDQN/weight/'
        os.mkdir(weight_path + name)
        # save weight
        pt_file = os.path.join(weight_path + name, 'actor_net.pt')
        torch.save(self.actor_net.state_dict(), pt_file)
        pt_file = os.path.join(weight_path + name, 'actor_target.pt')
        torch.save(self.actor_target.state_dict(), pt_file)
        pt_file = os.path.join(weight_path + name, 'param_net.pt')
        torch.save(self.param_net.state_dict(), pt_file)
        pt_file = os.path.join(weight_path + name, 'param_net_target.pt')
        torch.save(self.param_net_target.state_dict(), pt_file)
        print(name + 'net weights have been saved.')
        # save memory
        memory_path = 'PDQN/memory/'
        os.mkdir(memory_path + name)
        memory_file = os.path.join(memory_path + name, 'replay_buffer_backup.pkl')
        self.memory.backup(memory_file)
        print(name + 'replay buffer have been saved.')

    def load(self, name):
        weight_path = 'PDQN/weight/'
        # load weight
        pt_file = os.path.join(weight_path + name, 'actor_net.pt')
        self.actor_net.load_state_dict(torch.load(pt_file, map_location='cpu'))
        pt_file = os.path.join(weight_path + name, 'actor_target.pt')
        self.actor_target.load_state_dict(torch.load(pt_file, map_location='cpu'))
        pt_file = os.path.join(weight_path + name, 'param_net.pt')
        self.param_net.load_state_dict(torch.load(pt_file, map_location='cpu'))
        pt_file = os.path.join(weight_path + name, 'param_net_target.pt')
        self.param_net_target.load_state_dict(torch.load(pt_file, map_location='cpu'))
        print(name + 'net weights have been loaded.')
        memory_path = 'PDQN/memory/'
        memory_file = os.path.join(memory_path + name, 'replay_buffer_backup.pkl')
        self.memory.restore(memory_file)
        print(name + 'replay buffer have been loaded.')

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):
        if grad_type == "actions":
            max_p = self.action_max
            min_p = self.action_min
            rnge = self.action_range
        elif grad_type == "action_parameters":
            max_p = self.action_parameter_max
            min_p = self.action_parameter_min
            rnge = self.action_parameter_range
        else:
            raise ValueError("Unhandled grad_type: '" + str(grad_type) + "'")

        max_p = max_p.cpu()
        min_p = min_p.cpu()
        rnge = rnge.cpu()
        grad = grad.cpu()
        vals = vals.cpu()

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
            index = grad > 0
            grad[index] *= (index.float() * (max_p - vals) / rnge)[index]
            grad[~index] *= ((~index).float() * (vals - min_p) / rnge)[~index]

        return grad