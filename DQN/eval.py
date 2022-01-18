from config import *
if __name__ == '__main__':
    agent.load(path=cfg.model_path)
    rewards, ma_rewards = eval(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='eval', path=cfg.result_path)
    plot_rewards_cn(rewards, ma_rewards, tag='eval', env=cfg.env, algo=cfg.algo, path=cfg.result_path)