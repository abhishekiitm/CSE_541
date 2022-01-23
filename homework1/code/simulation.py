from bandit import Bandit
from algorithms import UCB, ETC, ThompsonSampling
from utils import aggregate_regrets, finalize_regrets

class Simulation(object):
    def __init__(self, n, T, means, n_sims):
        self.n = n
        self.T = T
        self.means = means
        self.n_sims = n_sims
        
    def simulate_ucb(self):
        count_aggregate, mean_aggregate, M2_aggregate = [[0 for i in range(self.T)] for _ in range(3)]
        for n_sim in range(self.n_sims):
            if n_sim%(self.n_sims/10)==0: print(f"Running simulation: {n_sim}/{self.n_sims}")
            bandit = Bandit(self.means)
            ucb_algo = UCB()
            regrets, arm_played, ucb, T_i = ucb_algo.play(self.T, self.means, bandit, 1)
            count_aggregate, mean_aggregate, M2_aggregate = aggregate_regrets(regrets, count_aggregate, mean_aggregate, M2_aggregate)
        mean_aggregate, var_aggregate = finalize_regrets(count_aggregate, mean_aggregate, M2_aggregate)
        return mean_aggregate, var_aggregate

    def simulate_thompson_sampling(self):
        count_aggregate, mean_aggregate, M2_aggregate = [[0 for i in range(self.T)] for _ in range(3)]
        for n_sim in range(self.n_sims):
            if n_sim%(self.n_sims/10)==0: print(f"Running simulation: {n_sim}/{self.n_sims}")
            bandit = Bandit(self.means)
            prior = [(0, 1) for i in range(self.n)]
            ts_algo = ThompsonSampling()
            regrets, arm_played, theta_hat_avgs, T_i = ts_algo.play(self.T, bandit, prior)
            count_aggregate, mean_aggregate, M2_aggregate = aggregate_regrets(regrets, count_aggregate, mean_aggregate, M2_aggregate)
        mean_aggregate, var_aggregate = finalize_regrets(count_aggregate, mean_aggregate, M2_aggregate)
        return mean_aggregate, var_aggregate

    def simulate_ect(self, m):
        count_aggregate, mean_aggregate, M2_aggregate = [[0 for i in range(self.T)] for _ in range(3)]
        for n_sim in range(self.n_sims):
            if n_sim%(self.n_sims/10)==0: print(f"Running simulation: {n_sim}/{self.n_sims}")
            bandit = Bandit(self.means)
            etc_algo = ETC()
            regrets, arm_played, ucb, T_i = etc_algo.play(self.T, m, self.means, bandit)
            count_aggregate, mean_aggregate, M2_aggregate = aggregate_regrets(regrets, count_aggregate, mean_aggregate, M2_aggregate)
        mean_aggregate, var_aggregate = finalize_regrets(count_aggregate, mean_aggregate, M2_aggregate)
        return mean_aggregate, var_aggregate