
import numpy as np
import matplotlib.pyplot as plt

class KArmedBandit(object): 
    def __init__(self, arms):
        self.bandits = []
        self.bandit_mean = 0
        self.bandit_variance = 1
        self.init_bandits(arms)

    def init_bandits(self, arms):
        for _ in range(arms):
            sampled_mean = np.random.normal(self.bandit_mean, 1.)
            bandit = Bandit(sampled_mean, self.bandit_variance)
            self.bandits.append(bandit)

        self.best_action = np.argmax([b.get_mean() for b in self.bandits])
        self.best_mean = self.bandits[self.best_action].get_mean()

        print("Best action: %d (Mean %.2f)" % (self.best_action, self.best_mean))

    def take_action(self, a):
        return

    def print_means(self):
        print(["%.2f" % bandit.get_mean() for bandit in self.bandits])

    def plot(self):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

        # Generate data
        data = [b.sample(100) for b in self.bandits]

        # Plot violin plot
        axes.violinplot(data, showmeans=False, showmedians=True)
        axes.set_title('%d-Armed Bandit' % len(self.bandits))

        # Adding horizontal grid lines
        axes.yaxis.grid(True)
        axes.set_xticks([y + 1 for y in range(len(data))])
        axes.set_xlabel('Action')
        axes.set_ylabel('Reward Distribution')

        # Add x-tick labels
        plt.setp(axes, xticks=[y + 1 for y in range(len(data))],
                 xticklabels=range(len(self.bandits)))
        plt.show()

class Bandit(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def sample(self, times=1):
        return np.random.normal(self.mean, self.var, times)

    def get_mean(self):
        return self.mean

    def get_var(self):
        return self.var

if __name__ == '__main__':
    kab = KArmedBandit(10)
    kab.print_means()
    kab.plot()
    