import numpy as np
import matplotlib.pyplot as plt

def basis(i, d):
    """
    returns standard basis vector e_i in R^d with 1 at idx i and 0 everywh
    """
    e = np.zeros(d)
    e[i] = 1
    return e

def G_optimal(aaT, lambda_t, X):
    """
    return G-optimal arm - argmax_{aj in X} ||aj||^2_{A(lambda)-1} 
    """
    n, d = X.shape
    A_t = np.sum(np.reshape(lambda_t, (n, 1, 1))*aaT, axis=0)
    A_t_inv = np.linalg.pinv(A_t)
    a2_A_t_inv = np.diagonal(np.matmul(np.matmul(X, A_t_inv), X.T))
    return np.argmax(a2_A_t_inv)

def f(aaT, lambda_hat, X):
    """
    returns value of f(lambda) - max_{aj in X} ||aj||^2_{A(lambda)-1} 
    """
    n, d = X.shape
    A_t = np.sum(np.reshape(lambda_hat, (n, 1, 1))*aaT, axis=0)
    A_t_inv = np.linalg.inv(A_t)
    a2_A_t_inv = np.diagonal(np.matmul(np.matmul(X, A_t_inv), X.T))
    return np.max(a2_A_t_inv)

def allocate_frank_wolfe(X, N):
    """
    returns a discrete allocation of size N for G-optimal design using greedy strategy
    """
    n, d = X.shape
    allocation = []
    
    # play first 2d arms uniformly at random
    #np.random.seed(0)
    initial_I = np.random.randint(0, n, size=2*d)
    for alloc in initial_I: allocation.append(alloc)

    lambda_t = np.zeros(n)
    I_counter = np.zeros(n)
    for idx in initial_I: 
        lambda_t[idx]+=1 
        I_counter[idx]+=1 
    lambda_t = lambda_t/2/d

    aaT = np.zeros((n, d, d))
    for idx in range(n): aaT[idx,:,:] = np.matmul(np.reshape(X[idx], (d, 1)), np.reshape(X[idx], (1, d)))

    for t in range(2*d+1, N+1):
        I_t = G_optimal(aaT, lambda_t, X)
        eta_t = 2/(t+1)
        lambda_t = (1-eta_t)*lambda_t + eta_t*basis(I_t, n)
        I_counter[I_t]+=1
        allocation.append(I_t)

    lambda_hat = I_counter/N
    return f(aaT, lambda_hat, X), allocation, lambda_hat


if __name__ == "__main__":
    d = 10
    N = 1000
    samples = 10
    a_vals = np.array([0, 0.5, 1, 2])
    j = np.arange(1, d+1)
    n_vals = 10+np.power(2, j)
    variance_vec = np.power(j[np.newaxis,:], -a_vals[:,np.newaxis])
    cov = np.diag(variance_vec[3])
    mean = np.zeros(d)

    f_lambda_hat_mean_array = np.zeros((len(a_vals), len(n_vals)))
    f_lambda_hat_stdev_array = np.zeros((len(a_vals), len(n_vals)))

    for idx_a, a in enumerate(a_vals):
        for idx_n, n in enumerate(n_vals):
            cov = np.diag(variance_vec[idx_a])
            #np.random.seed(0)

            f_lambda_hats = [0]*samples
            for j in range(samples):
                X = np.random.multivariate_normal(mean, cov, size=n)
                f_lambda_hat, allocation, lbda = allocate_frank_wolfe(X, N)
                f_lambda_hats[j] = f_lambda_hat
            
            f_lambda_hat_mean_array[idx_a, idx_n] = np.mean(f_lambda_hats)
            f_lambda_hat_stdev_array[idx_a, idx_n] = np.std(f_lambda_hats)

    colors = ['b', 'g', 'r', 'm', 'c', 'tab:brown', 'tab:grey', 'black']

    #plt.plot(standard_bound, color=colors[0], label='standard bound')
    plt.xscale("log")
    for i, a_val in enumerate(a_vals):
        plt.plot(n_vals, f_lambda_hat_mean_array[i], color=colors[i], label=f"f(lambda_hat) for a = {a_val}")
        plt.errorbar(n_vals, f_lambda_hat_mean_array[i], f_lambda_hat_stdev_array[i], linestyle='None', fmt='o', color=colors[i], capsize=5, alpha=0.5, barsabove=True)

    plt.legend(loc='upper right')
    plt.xlabel('n')
    plt.ylabel('f(lambda_hat) for each setting')
    plt.show()

    pass
