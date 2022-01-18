import gym
import datetime
import torch

from agent import DQN
from common.utils import make_dir, save_results
from common.plot import plot_rewards_cn

curr_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

class DQNConfig:
    def __init__(self):
        self.algo = 'DQN'
        self.env = 'LunarLander-v2'
        self.result_path = './outputs/' + '/results/'
        self.model_path = './outputs/' + '/models/'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = 0.95
        self.epsilon_start = 0.90
        self.epsilon_end = 0.01
        self.epsilon_decay = 5000
        self.batch_size = 64
        self.hidden_dim = 256
        self.capacity = 100000
        self.lr = 0.0001
        self.train_eps = 200
        self.eval_eps = 5
        self.target_update = 4

def env_agent_config(cfg, seed=1):
    env = gym.make(cfg.env)
    env.seed(seed)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQN(n_states, n_actions, cfg)
    return env, agent

def train(cfg, env, agent):
    print('开始训练！')
    print(f'环境:{cfg.env}, 算法:{cfg.algo}, 设备:{cfg.device}')
    rewards = []
    ma_rewards = []
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            if done:
                break
        if (i_ep+1) % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        if (i_ep+1) % 10 == 0:
            print('回合: {}/{}, 奖励: {}'.format(i_ep+1, cfg.train_eps, ep_reward))
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('完成训练！')
    return rewards, ma_rewards

def eval(cfg, env, agent):
    print('开始测试！')
    print(f'环境: {cfg.env}, 算法: {cfg.algo}, 设备: {cfg.device}')
    rewards = []
    ma_rewards = []
    for i_ep in range(cfg.eval_eps):
        ep_reward = 0
        state = env.reset()
        while True:
            action = agent.predict(state)
            next_state, reward, done, _ = env.step(action)
            env.render()
            state = next_state
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print(f'回合: {i_ep+1}/{cfg.eval_eps}, 奖励:{ep_reward}')
    print('完成测试！')
    return rewards, ma_rewards

cfg = DQNConfig()
env, agent = env_agent_config(cfg)
