import numpy as np
import math

class UCB(object):
    def __init__(self):
        pass

    def play(self, T, means, bandit, alpha=1):
        def play_arm(i, t):
            theta_hat_i = bandit.pull_arm(i)
            theta_hat_sums[i] += theta_hat_i
            T_i[i] += 1
            ucb[i] = theta_hat_sums[i]/T_i[i] + alpha*math.sqrt(2*math.log(2*n*T*T)/T_i[i])
            regrets[t] = bandit.get_regret()
            arm_played[t] = i

        n = len(means)
        theta_hat_sums = [0 for i in means]
        T_i = [0 for i in means]
        ucb = [float("inf") for i in means]
        regrets = [0 for t in range(T)]
        arm_played = [0 for t in range(T)]

        for i in range(len(means)):
            play_arm(i, i)

        for t in range(n, T):
            I_t = np.argmax(ucb)
            play_arm(I_t, t)
            
        return regrets, arm_played, ucb, T_i

class ETC(object):
    def __init__(self):
        pass

    def play(self, T, m, means, bandit):
        def play_arm(i, t):
            theta_hat_i = bandit.pull_arm(i)
            theta_hat_sums[i] += theta_hat_i
            T_i[i] += 1
            theta_hat_avgs[i] = theta_hat_sums[i]/T_i[i]
            regrets[t] = bandit.get_regret()
            arm_played[t] = i

        n = len(means)
        theta_hat_sums = [0 for i in means]
        theta_hat_avgs = [0 for i in means]
        T_i = [0 for i in means]
        regrets = [0 for t in range(T)]
        arm_played = [0 for t in range(T)]

        for t in range(T):
            if t<m*n:
                i = t%n
                play_arm(i, t)
            else:
                if t==m*n: I_t = np.argmax(theta_hat_avgs)
                play_arm(I_t, t)

        return regrets, arm_played, theta_hat_avgs, T_i

class ThompsonSampling(object):
    def __init__(self):
        pass

    def sample_theta(self, distribution):
        sample = [np.random.normal(loc=mean, scale=math.sqrt(variance)) for mean, variance in distribution]
        return sample
        
    def compute_posterior(self, X, mu0, var0):
        var = 1
        var_posterior = var*var0/(var+var0)
        mean_posterior = var_posterior*(mu0/var0 + X/var)
        return (mean_posterior, var_posterior)


    def play(self, T, bandit, prior):
        def play_arm(i, t):
            theta_hat_i = bandit.pull_arm(i)
            theta_hat_sums[i] += theta_hat_i
            T_i[i] += 1
            theta_hat_avgs[i] = theta_hat_sums[i]/T_i[i]
            regrets[t] = bandit.get_regret()
            arm_played[t] = i
            prior[i] = self.compute_posterior(theta_hat_i, prior[i][0], prior[i][1])

        theta_hat_sums = [0 for i in prior]
        theta_hat_avgs = [0 for i in prior]
        T_i = [0 for i in prior]
        regrets = [0 for t in range(T)]
        arm_played = [0 for t in range(T)]

        for t in range(T):
            sample = self.sample_theta(prior)
            I_t = np.argmax(sample)
            play_arm(I_t, t)

        return regrets, arm_played, theta_hat_avgs, T_i
            
