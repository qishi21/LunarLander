
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPGConfig:
    def __init__(self):
        self.env = 'LunarLander-v2'
        self.hidden_dim_1 = 400
        self.hidden_dim_2 = 200



class DDPGAgent:
    def __init__(self):
        pass


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        pass


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        pass