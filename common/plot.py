import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei Mono']

def plot_rewards_cn(rewards, marewards, tag, env, algo, path, save=True):
    plt.figure()
    plt.title('{}环境下{}算法的学习曲线'.format(env, algo))
    plt.xlabel('回合数')
    plt.plot(rewards)
    plt.plot(marewards)
    plt.legend(('奖励', '滑动平均奖励'), loc='best')
    if save:
        plt.savefig(path+f'{tag}_rewards_curve_cn')
    plt.show()