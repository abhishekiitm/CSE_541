import numpy as np
from utils import aggregate_regrets, finalize_regrets, plot
from linear_bandit_algorithms import Elimination_algorithm, UCB_algorithm, Thompson_sampling

def generate_features(n, d):
    X = np.concatenate( ( np.linspace(0, 1, 50), 0.25+0.01*np.random.randn(250) ) , 0)
    X = np.sort(X)

    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n) :
            K[i, j] = 1+ min(X[i], X[j])
    e, v = np.linalg.eigh(K) # eigenvalues are increasing in order
    
    Phi = np.real(v@np.diag(np.sqrt(np.abs(e))))[:,(n-d)::]

    return Phi, np.expand_dims(X, axis=1)

def simulation():
    n_sims = 100
    T = 40000
    n = 300
    d = 30

    sim_types = [
        ('ELIM', "Elimination Algorithm with G-optimal design(Frank Wolfe algo.)"),
        ('UCB', "UCB Algorithm"),
        ('TS', "Thompson Sampling"),
    ]

    Phi, X = generate_features(n, d)

    mean_result, var_result, labels = [[0 for i in range(len(sim_types))] for _ in range(3)]

    for i, (key, label) in enumerate(sim_types):
        labels[i] = label
        count_aggregate, mean_aggregate, M2_aggregate = [[0 for i in range(T)] for _ in range(3)]
        for n_sim in range(n_sims):
            if n_sim%(n_sims/10)==0: print(f"Running simulation for {key}: {n_sim}/{n_sims}")
            if key == 'ELIM':
                regret, I = Elimination_algorithm(T, Phi, X)
            if key == 'UCB':
                regret, I = UCB_algorithm(T, Phi, X)
            if key == 'TS':
                regret, I = Thompson_sampling(T, Phi, X)
            count_aggregate, mean_aggregate, M2_aggregate = aggregate_regrets(regret, count_aggregate, mean_aggregate, M2_aggregate)
        
        mean_result[i], var_result[i] = finalize_regrets(count_aggregate, mean_aggregate, M2_aggregate)

    plot(mean_result, var_result, labels, T)

if __name__=="__main__":
    simulation()