from copyreg import pickle
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

def append(arr1, arr2):
    if arr1 is None: return arr2
    return np.r_[arr1, arr2]

def load_pickle(filepath):
    with open(filepath, 'rb') as inp:
        pca = pickle.load(inp)
    return pca

def load_train_downsampled(filepath):
    downsampled_data = np.loadtxt(filepath, delimiter=",")
    sampled_images, sampled_labels = downsampled_data[:,:-1], np.squeeze(downsampled_data[:,-1:])

    return sampled_images, sampled_labels

def rescale_norm(X):
    X_norm_vec = np.linalg.norm(X, axis=1, keepdims=True)
    X = X/X_norm_vec
    return X

def preview(data, labels, i):
    img = data[i].reshape((28,28))
    plt.imshow(img, cmap="Greys")
    plt.title(labels[i])
    plt.show()

def plot(mean_aggregate, var_aggregate, labels, T):
    mean_aggregate = [np.array(item) for item in mean_aggregate]
    std_aggregate = [np.sqrt(item) for item in var_aggregate]

    x_error = np.arange(0, T+1, T/5, dtype=np.int32)
    x_error[-1] = x_error[-1]-1
    
    for i in range(len(mean_aggregate)):
        colors = ['b', 'g', 'r', 'm', 'c', 'k', 'tab:grey']
        y_error = mean_aggregate[i][x_error]
        #e = std_aggregate[i][x_error]

        time_series_df = pd.DataFrame(mean_aggregate[i])

        plt.plot(time_series_df, linewidth=1, label=labels[i], color=colors[i]) #mean curve.
        #plt.errorbar(x_error, y_error, e, linestyle='None', fmt='o', color=colors[i], capsize=5, alpha=0.5, barsabove=True)
    plt.legend(loc='upper left')
    plt.ylabel("Regret at time step t")
    plt.xlabel("Time step t")
    plt.title("Comparison of Regret at time t, (algorithm run only for one simulation)")
    plt.rcParams["figure.figsize"] = (30,6)
    plt.savefig('results/plot.png')
    plt.show()