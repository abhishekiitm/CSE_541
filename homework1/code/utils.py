from turtle import color
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)

# Retrieve the mean, variance and sample variance from an aggregate
def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sampleVariance)


def aggregate_regrets(regrets, count_aggregate, mean_aggregate, M2_aggregate):
    for i in range(len(regrets)):
        count_aggregate[i], mean_aggregate[i], M2_aggregate[i] = \
            update((count_aggregate[i], mean_aggregate[i], M2_aggregate[i]), regrets[i])
    
    return count_aggregate, mean_aggregate, M2_aggregate

def finalize_regrets(count_aggregate, mean_aggregate, M2_aggregate):
    variance_aggregate = [0 for i in mean_aggregate]
    for i in range(len(mean_aggregate)):
        mean_aggregate[i], variance_aggregate[i], _ = finalize((count_aggregate[i], mean_aggregate[i], M2_aggregate[i]))

    return mean_aggregate, variance_aggregate


def plot(mean_aggregate, var_aggregate, labels, T):
    mean_aggregate = [np.array(item) for item in mean_aggregate]
    std_aggregate = [np.sqrt(item) for item in var_aggregate]

    x_error = np.arange(0, T+1, T/5, dtype=np.int32)
    x_error[-1] = x_error[-1]-1
    
    for i in range(len(mean_aggregate)):
        colors = ['b', 'g', 'r', 'm', 'c', 'k', 'tab:grey']
        y_error = mean_aggregate[i][x_error]
        e = std_aggregate[i][x_error]

        time_series_df = pd.DataFrame(mean_aggregate[i])

        plt.plot(time_series_df, linewidth=1, label=labels[i], color=colors[i]) #mean curve.
        plt.errorbar(x_error, y_error, e, linestyle='None', fmt='o', color=colors[i], capsize=5, alpha=0.5, barsabove=True)
    plt.legend(loc='upper left')
    plt.ylabel("Average Regret at time step t")
    plt.xlabel("Time step t")
    plt.title("Comparison of Average Regret at time t, bars denoted 1 standard deviation around the mean")
    plt.savefig('results/plot.png')
    plt.show()
    