import numpy as np
from g_optimal_design import allocate_frank_wolfe
import math

def f(x):
    return -x **2 + x*np.cos(8*x) + np.sin(15*x)

def observe(X, idx) :
    return f(X[idx]) + np.random.randn(len(idx), 1)


def Elimination_algorithm(T, Phi, X):
    tau = 100
    delta = 1/T
    gamma = 1
    U = 1
    n, d = Phi.shape
    V_0, S_0 = gamma * np.eye(d), np.zeros((d, 1))
    Phi_k_hat = Phi
    X_k_hat = X
    N = 1000

    Y = np.zeros(T)
    I = np.zeros(T)

    V_k, S_k = V_0, S_0
    det_V_0 = np.linalg.det(V_0)
    for k in range(1, math.floor(T/tau)):
        if Phi_k_hat.shape[0] == 1: break
        # get G-optimal allocation
        _, allocation, lbda_k = allocate_frank_wolfe(Phi_k_hat, N)

        # sample and observe based on the G-optimal allocation
        idx = np.random.choice(np.arange(Phi_k_hat.shape[0]), size=tau, p=lbda_k)
        y = observe(X_k_hat, idx)
        Y[(k-1)*tau:(k-1)*tau+tau] = np.squeeze(y)
        I[(k-1)*tau:(k-1)*tau+tau] = len(Phi_k_hat)

        # update V_k, S_k, theta_k
        V_k = V_k + Phi_k_hat[idx].T @ Phi_k_hat[idx]
        S_k = S_k + np.sum((y.T)*(Phi_k_hat[idx].T), axis=1, keepdims=True)
        V_k_inv = np.linalg.inv(V_k)
        theta_k = V_k_inv @ S_k
        beta_k = np.sqrt(gamma)*U + np.sqrt(2*np.log(1/delta) + np.log( np.abs(np.linalg.det(V_k)/det_V_0) ) )

        # Eliminate arms
        i_max = np.argmax(Phi_k_hat @ theta_k)
        keep_indices = np.squeeze(((Phi_k_hat[[i_max]] - Phi_k_hat) @ theta_k) <= beta_k*\
            np.sum(((Phi_k_hat[[i_max]] - Phi_k_hat) @ V_k_inv) * (Phi_k_hat[[i_max]] - Phi_k_hat), axis=1, keepdims=True))
        Phi_k_hat = Phi_k_hat[keep_indices]
        X_k_hat = X_k_hat[keep_indices]
    
    # play remaining arms
    rem = T-(k-1)*tau
    if rem > 0:
        I[-rem:] = 1 
        y = np.random.randn(rem) + f(X_k_hat[0, 0])
        Y[-rem:] = y
        I[-rem:] = 1
    
    regret = max(f(X)) - Y
    return np.cumsum(regret), I


def UCB_algorithm(T, Phi, X):
    delta = 1/T
    gamma = 1
    U = 1
    n, d = Phi.shape
    V_0, S_0 = gamma * np.eye(d), np.zeros((d, 1))
    det_V_0 = np.linalg.det(V_0)

    Y = np.zeros(T)
    I = np.zeros(T)

    V_t, S_t = V_0, S_0
    for t in range(T):
        #print(f"iteration {t} of UCB")
        beta_t = np.sqrt(gamma)*U + np.sqrt(2*np.log(1/delta) + np.log( np.abs(np.linalg.det(V_t)/det_V_0) ) )
        V_t_inv = np.linalg.inv(V_t)
        theta_t = V_t_inv @ S_t
        i_t = np.argmax( Phi @ theta_t + beta_t*np.sqrt(np.sum(Phi @ V_t_inv * Phi, axis=1, keepdims=True)) )
        
        # pull arm and observe
        Y[t] = observe(X, [i_t])
        I[t] = i_t

        # update V_t, S_t
        V_t = V_t + Phi[[i_t]].T @ Phi[[i_t]]
        S_t = S_t + Y[t]*(Phi[[i_t]].T)

    regret = max(f(X)) - Y
    return np.cumsum(regret), I


def Thompson_sampling(T, Phi, X):
    gamma = 1
    n, d = Phi.shape
    V_0, S_0 = gamma * np.eye(d), np.zeros((d, 1))

    Y = np.zeros(T)
    I = np.zeros(T)

    V_t, S_t = V_0, S_0
    for t in range(T):
        V_t_inv = np.linalg.inv(V_t)
        theta_t = V_t_inv @ S_t
        theta_sample = np.random.multivariate_normal(np.squeeze(theta_t), V_t_inv)
        i_t = np.argmax(Phi @ theta_sample)

        # pull arm and observe
        Y[t] = observe(X, [i_t])
        I[t] = i_t

        # update V_t, S_t
        V_t = V_t + Phi[[i_t]].T @ Phi[[i_t]]
        S_t = S_t + Y[t]*(Phi[[i_t]].T)

    regret = max(f(X)) - Y
    return np.cumsum(regret), I


