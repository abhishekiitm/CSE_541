import numpy as np

class Bandit(object):
    """
    Implements a K  armed Bandit
    """
    def __init__(self, means):
        self.means = means
        self.K = len(means)
        self.optimal_mean = max(means)
        self.regret = 0
        self.last_regret = 0

    def K(self):
        return self.K
    
    def pull_arm(self, i):
        X_i = np.random.normal(loc=self.means[i], scale=1)
        self.regret += self.optimal_mean - X_i
        self.last_regret = self.optimal_mean - X_i
        return X_i

    def get_regret(self):
        return self.regret

    def latest_regret(self):
        return self.last_regret