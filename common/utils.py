from pathlib import Path
import numpy as np

def make_dir(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_results(rewards, ma_rewards, tag, path):
    np.save(path + '{}_rewards.npy'.format(tag), rewards)
    np.save(path + '{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('结果保存完毕！')
