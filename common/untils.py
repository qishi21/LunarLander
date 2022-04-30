import random
import numpy as np


# 经验回放池
class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.index = 0

    def store_transition(self, state, action, reward, next_state, done):
        if self.index < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.index % self.capacity] = (state, action, reward, next_state, done)
        self.index += 1

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        return np.array(state_batch), np.array(action_batch), np.array(reward_batch), np.array(next_state_batch), np.array(done_batch)

    def __len__(self):
        return len(self.buffer)


# OU噪声
class OUActionNoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta*(self.mu-self.x_prev)*self.dt + self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 else np.zeros_like(self.mu)
