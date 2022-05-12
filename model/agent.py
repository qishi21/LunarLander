import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from common.untils import ReplayBuffer, OUActionNoise
from model.DDPGNetworks import ActorNetwork, CriticNetwork


class DDPGAgent:
    def __init__(self, cfg, state_dim, action_dim):
        self.tau = cfg.tau
        self.batch_size = cfg.batch_size
        self.alpha = cfg.alpha
        self.beta = cfg.beta
        self.gamma = cfg.gamma
<<<<<<< Updated upstream:model/agent.py
<<<<<<< Updated upstream:agent.py
=======
=======
>>>>>>> Stashed changes:agent.py
        self.device = cfg.device
>>>>>>> Stashed changes:model/agent.py

        self.memory = ReplayBuffer(cfg.capacity)  # 经验回放池
        self.noise = OUActionNoise(mu=np.zeros(action_dim))  # OU噪声

        # actor_network
        self.actor = ActorNetwork(state_dim, action_dim, cfg.fc1_dim, cfg.fc2_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.alpha)
        self.actor_target = ActorNetwork(state_dim, action_dim, cfg.fc1_dim, cfg.fc2_dim)

        # critic_network
        self.critic = CriticNetwork(state_dim, action_dim, cfg.fc1_dim, cfg.fc2_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.beta, weight_decay=0.01)
<<<<<<< Updated upstream:agent.py
        self.critic_target = CriticNetwork(state_dim, action_dim, cfg.fc1_dim, cfg.fc2_dim)
=======
        self.critic_target = CriticNetwork(state_dim, action_dim, cfg.fc1_dim, cfg.fc2_dim).to(cfg.device)
<<<<<<< Updated upstream:model/agent.py
>>>>>>> Stashed changes:model/agent.py
=======
>>>>>>> Stashed changes:agent.py

        # 软更新
        self.update_network_parameters(tau=1)

    # 选择动作，用于训练
    def choose_action(self, state):
        self.actor.eval()

        state = torch.tensor(np.array([state]), dtype=torch.float)
        mu = self.actor(state)
        mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float)

        self.actor.train()
        return mu_prime.detach().numpy()[0]

    # 预测动作，用于评估
    def predict(self, state):
        self.actor.eval()
        state = torch.tensor(np.array([state]), dtype=torch.float)
        mu = self.actor(state)
        self.actor.train()
        return mu.detach().numpy()[0]

    # 往经验池添加数据
    def push(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    # agent学习
    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones)

        actions_target = self.actor_target(next_states)
        next_state_critic_value = self.critic_target(next_states, actions_target)
        state_critic_value = self.critic(states, actions)

        next_state_critic_value[dones] = 0.
        next_state_critic_value = next_state_critic_value.view(-1)

        target = rewards + self.gamma * next_state_critic_value
        target = target.view(self.batch_size, 1)

        self.actor_optimizer.zero_grad()
        actor_loss = -self.critic(states, self.actor(states))
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss = nn.MSELoss()(target, state_critic_value)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.update_network_parameters()

    # soft update
    def update_network_parameters(self, tau=None):
        if not tau:
            tau = self.tau

        actor_params_dict = OrderedDict(self.actor.named_parameters())
        critic_params_dict = OrderedDict(self.critic.named_parameters())
        actor_target_params_dict = OrderedDict(self.actor_target.named_parameters())
        critic_target_params_dict = OrderedDict(self.critic_target.named_parameters())

        for name in critic_params_dict:
            critic_params_dict[name] = tau*critic_params_dict[name].clone() + (1-tau)*critic_target_params_dict[name].clone()

        for name in actor_params_dict:
<<<<<<< Updated upstream:model/agent.py
<<<<<<< Updated upstream:agent.py
            actor_params_dict[name] = tau * actor_params_dict[name].clone() + (1-tau)*actor_target_params_dict[name].clone()
=======
            actor_params_dict[name] = tau*actor_params_dict[name].clone() + (1-tau)*actor_target_params_dict[name].clone()
>>>>>>> Stashed changes:model/agent.py
=======
            actor_params_dict[name] = tau*actor_params_dict[name].clone() + (1-tau)*actor_target_params_dict[name].clone()
>>>>>>> Stashed changes:agent.py

        self.critic_target.load_state_dict(critic_params_dict)
        self.actor_target.load_state_dict(actor_params_dict)

    # 保存模型
    def save_models(self, model_dir='./params/last_model'):
        self.actor.save_model('actor', model_dir)
        self.actor_target.save_model('actor_target', model_dir)
        self.critic.save_model('critic', model_dir)
        self.critic_target.save_model('critic_target', model_dir)

    # 载入模型
    def load_models(self, model_dir='./params/last_model'):
        self.actor.load_model('actor', model_dir)
        self.actor_target.load_model('actor_target', model_dir)
        self.critic.load_model('critic', model_dir)
        self.critic_target.load_model('critic_target', model_dir)
