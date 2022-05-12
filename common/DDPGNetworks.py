import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# actor_network
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim, fc2_dim):
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)

        self.bn1 = nn.LayerNorm(fc1_dim)
        self.bn2 = nn.LayerNorm(fc2_dim)

        self.mu = nn.Linear(fc2_dim, action_dim)

        # 初始化参数
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        # 初始化参数
        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        # 初始化参数
        f3 = 3e-3
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.mu(x)
        x = torch.tanh(x)
        return x

    def save_model(self, model_name, model_dir='./models'):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.state_dict(), f'{model_dir}/DDPG_{model_name}.pth')

    def load_model(self, model_name, model_dir='./models'):
        self.load_state_dict(torch.load(f'{model_dir}/DDPG_{model_name}.pth'))


# critic_network
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)

        self.bn1 = nn.LayerNorm(fc1_dim)
        self.bn2 = nn.LayerNorm(fc2_dim)

        self.action_value = nn.Linear(action_dim, fc2_dim)

        self.q = nn.Linear(self.fc2_dim, 1)

        # 初始化参数
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc2.bias.data.uniform_(-f1, f1)

        # 初始化参数
        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        # 初始化参数
        f3 = 1. / np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f3, f3)
        self.action_value.bias.data.uniform_(-f3, f3)

        # 初始化参数
        f4 = 3e-3
        self.q.weight.data.uniform_(-f4, f4)
        self.q.bias.data.uniform_(-f4, f4)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        action_value = self.action_value(action)
        state_action_value = F.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_model(self, model_name, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.state_dict(), f'{model_dir}/DDPG_{model_name}.pth')

    def load_model(self, model_name, model_dir):
        self.load_state_dict(torch.load(f'{model_dir}/DDPG_{model_name}.pth'))
