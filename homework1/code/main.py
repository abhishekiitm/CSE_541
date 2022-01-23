from simulation import Simulation
from utils import plot
import math

if __name__=="__main__":
    n = 40
    means = [1]
    means.extend([1-1/math.sqrt(i-1) for i in range(2, n+1)])
    T = 30000
    n_sims = 100

    sim_types = [
        ('ECT', "ECT ; m = 100", 100),
        ('ECT', "ECT ; m = 200", 200),
        ('ECT', "ECT ; m = 300", 300),
        ('UCB', "UCB", None),
        ('TS', "Thompson Sampling", None),
    ]

    mean_aggregate, var_aggregate, labels = [[0 for i in range(len(sim_types))] for _ in range(3)]


    simulation = Simulation(n, T, means, n_sims)
    for i, (key, label, m) in enumerate(sim_types):
        labels[i] = label
        if key == 'ECT': mean_aggregate[i], var_aggregate[i] = simulation.simulate_ect(m)
        elif key == 'UCB': mean_aggregate[i], var_aggregate[i] = simulation.simulate_ucb()
        elif key == 'TS': mean_aggregate[i], var_aggregate[i] = simulation.simulate_thompson_sampling()
        else:
            print('Invalid Key type')

    plot(mean_aggregate, var_aggregate, labels, T)




