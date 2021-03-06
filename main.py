import torch
import logging
import os
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model.agent import DDPGAgent

TRAIN_CHECKPOINT = False  # 是否进行训练
LOAD_MODEL_CHECKPOINT = True  # 是否载入模型
RENDER_CHECKPOINT = True  # 是否显示动画（仅在评估时有效）
LOGGER_CHECKPOINT = False  # 是否将运行log写入文件
TENSORBOARD_CHECKPOINT = False  # 是否开启tensorboard


# 参数配置
class DDPGConfig:
    def __init__(self):
        self.env_name = 'LunarLanderContinuous-v2'  # 环境名称
        self.algo = 'DDPG'  # 算法名称
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.alpha = 1e-4  # actor_network学习率
        self.beta = 1e-3  # critic_network学习率
        self.gamma = 0.99  # 折扣系数
        self.tau = 0.05  # 软更新系数 5e-3
        self.batch_size = 64  # 每次从经验回放池中抽取的样本数量
        self.capacity = 1e6  # 经验回放池的容量

        self.fc1_dim = 300  # 隐藏层1的维度
        self.fc2_dim = 200  # 隐藏层2的维度

        self.train_eps = 1000  # 训练的幕数
        self.eval_eps = 20  # 评估的幕数

        # 记录最大的奖励值，便于保存最优的模型
        self.max_reward = 0

        # tensorboard
        if TENSORBOARD_CHECKPOINT:
            self.writer = SummaryWriter()

        # logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('%(asctime)20s|%(levelname)s|%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.console_handler)
        if LOGGER_CHECKPOINT:
            self.file_handler = logging.FileHandler(filename=f'{self.env_name}_temp.log')
            self.file_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.file_handler)


# 配置环境和agent
def env_agent_config(cfg):
    env = gym.make(cfg.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPGAgent(cfg, state_dim, action_dim)
    if LOAD_MODEL_CHECKPOINT:
        agent.load_models('./params/best_model')
        cfg.logger.info('Load best model.')

    return env, agent


# 训练
def train(cfg, env, agent):
    cfg.logger.info('Start training.')
    cfg.logger.info(f'env:{cfg.env_name}, algo:{cfg.algo}, device:{cfg.device}')
    history_reward = []

    for ep in range(cfg.train_eps):
        steps = 0
        ep_reward = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            steps += 1
            next_state, reward, done, _ = env.step(action)
            shaping_reward = reward - abs(next_state[0])*10
            if done:
                shaping_reward += (state[6]+state[7])*10
            agent.push(state, action, shaping_reward, next_state, done)
            agent.learn()
            ep_reward += reward
            state = next_state

        if TENSORBOARD_CHECKPOINT:
            cfg.writer.add_scalar('train/reward', ep_reward, ep)
            cfg.writer.add_scalar('train/steps', steps, ep)
        cfg.logger.info(f'episode:{ep+1:>3}/{cfg.train_eps}|steps:{steps:>4}|ep_reward:{round(ep_reward, 2)}')
        history_reward.append(ep_reward)
        mean_reward = np.mean(history_reward[-100:])
        if mean_reward > cfg.max_reward:
            model_dir = 'params/best_model'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            agent.save_models(model_dir)
            cfg.logger.info(f'Save best model. 前{min(len(history_reward), 100)}幕的平均reward为{np.round(mean_reward, 2)}.')
            cfg.max_reward = mean_reward

    agent.save_models()
    cfg.logger.info('Save last model.')
    cfg.logger.info('Complete training.')
    cfg.logger.info('='*43)


# 评估
def eval(cfg, env, agent):
    cfg.logger.info('Start evaluation.')
    cfg.logger.info(f'env:{cfg.env_name}, algo:{cfg.algo}, device:{cfg.device}')

    history_reward = []
    for ep in range(cfg.eval_eps):
        ep_reward = 0
        steps = 0
        state = env.reset()
        if RENDER_CHECKPOINT:
            env.render()
        done = False
        while not done:
            action = agent.predict(state)
            steps += 1
            next_state, reward, done, _ = env.step(action)
            if RENDER_CHECKPOINT:
                env.render()
            agent.push(state, action, reward, next_state, done)
            ep_reward += reward
            state = next_state
        history_reward.append(ep_reward)
        if TENSORBOARD_CHECKPOINT:
            cfg.writer.add_scalar('eval/reward', ep_reward, ep)
            cfg.writer.add_scalar('eval/steps', steps, ep)
        cfg.logger.info(f'episode:{ep+1:>3}/{cfg.eval_eps}|steps:{steps:>4}|ep_reward:{round(ep_reward, 2)}')

    cfg.logger.info(f'共计{cfg.eval_eps}幕，平均reward为{np.round(np.mean(history_reward), 2)}.')
    cfg.logger.info('Complete evaluation.')
    cfg.logger.info('='*43)


if __name__ == '__main__':
    cfg = DDPGConfig()
    env, agent = env_agent_config(cfg)

    if TRAIN_CHECKPOINT:
        try:
            train(cfg, env, agent)
        except KeyboardInterrupt:
            agent.save_models()
            cfg.logger.info('KeyboardInterrupt.')
            cfg.logger.info('Save last model.')
            cfg.logger.info('='*43)
            if TENSORBOARD_CHECKPOINT:
                cfg.writer.close()
    else:
        eval(cfg, env, agent)

    if TENSORBOARD_CHECKPOINT:
        cfg.writer.close()
