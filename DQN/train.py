from config import *
from common.utils import make_dir, save_results
from common.plot import plot_rewards_cn

if __name__ == '__main__':
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    plot_rewards_cn(rewards, ma_rewards, tag='train', env=cfg.env, algo=cfg.algo, path=cfg.result_path)