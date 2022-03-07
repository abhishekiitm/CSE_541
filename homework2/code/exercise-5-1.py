import math
import numpy as np
import matplotlib.pyplot as plt

def fixed_time_tail_bound(delta, t):
    num = 2*np.log(2/delta)
    den = t
    return np.sqrt(num/den)

def sequential_tail_bound(delta, t, var):
    temp = (1+1/t/var)*(2*np.log(1/delta) + np.log(t*var + 1))/t
    return np.sqrt(temp)

if __name__ == "__main__":
    delta = 0.05  # confidence level
    variances = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]

    var = variances[0]

    T_start = 10
    T_end = 10000000
    t_arr = np.arange(T_start, T_end+1, step=5)

    fixed_time_bound = fixed_time_tail_bound(delta, t_arr)

    colors = ['b', 'g', 'r', 'm', 'c', 'tab:brown', 'tab:grey', 'black']

    plt.yscale("log")
    plt.xscale("log")
    #plt.plot(standard_bound, color=colors[0], label='standard bound')
    for i, var in enumerate(variances):
        sequential_bound = sequential_tail_bound(delta, t_arr, var)
        seq_to_standard_ratio = sequential_bound / fixed_time_bound
        plt.plot(t_arr, seq_to_standard_ratio, color=colors[i], label=r"seq. bound with $\sigma^2$"+f" = {var}")

    plt.legend(loc='upper right')
    plt.xlabel('t')
    plt.ylabel('ratio of sequential bound to fixed time bound')
    plt.title(r'Ratio of anytime confidence bounds to fixed time bounds for different $\sigma^2$')
    plt.show()
