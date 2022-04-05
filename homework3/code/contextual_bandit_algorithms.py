import math
import numpy as np
from utils import load_train_downsampled, load_pickle, preview, rescale_norm, append, plot
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import time

def basis(i, d):
    """
    returns standard basis vector e_i in R^d with 1 at idx i and 0 everywh
    """
    e = np.zeros(d)
    e[i] = 1
    return e

def generate_Phi(C, a, k):
    tau, d = len(a), C.shape[1]
    Phi = np.zeros((tau, d*k))
    for j, action in enumerate(a):
        e_a = basis(action, k).reshape(k, 1)
        c = C[j].reshape(len(C[j]), 1)
        phi_c_a = ( c @ e_a.T ).flatten()
        Phi[j] = phi_c_a
        
    return Phi

def select_best_arm(c_t, theta_hat, k):
    Phi_c_t = np.einsum("i,jk->kij", c_t, np.eye(k))
    Phi_c_t = Phi_c_t.reshape(k, -1)
    return np.argmax(Phi_c_t @ theta_hat)

def commit(C, theta_hat, tau, k):
    T, d = C.shape
    commit_plays = []
    for t in range(tau, T):
        commit_plays.append( select_best_arm(C[t], theta_hat, k) )
    
    return commit_plays

def ETC_world(X_train, y_train):
    T, d = X_train.shape
    k = len(np.unique(y_train))

    # calculate tau
    tau = math.ceil(T**(2/3) * (2*k*(k*d+1)*math.log(2))**(1/3) )

    # explore to get theta_hat
    explore_plays = np.random.choice(np.arange(k), size=tau)
    Phi = generate_Phi(X_train_pca_norm, explore_plays, k)
    r = np.where(y_train[:tau] == explore_plays, 1, 0).reshape(tau, 1)
    theta_hat = (np.linalg.pinv(Phi.T @ Phi) @ Phi.T) @ r

    # play arg max based on theta_hat
    commit_plays = commit(X_train_pca_norm, theta_hat, tau, k)

    # calculate regret
    arms_played = np.r_[explore_plays, commit_plays]
    regret = np.where(arms_played == y_train, 0, 1)

    return np.cumsum(regret)

def ETC_bias(X_train, y_train):
    T, d = X_train.shape
    k = len(np.unique(y_train))

    # calculate tau
    tau = math.ceil(T**(2/3) * (2*k*(k*d+1)*math.log(2))**(1/3) )

    # explore to generate data for logistic classifier
    explore_plays = np.random.choice(np.arange(k), size=tau)
    indices = np.where(explore_plays == y_train[:tau])[0]
    X = X_train[indices]
    y = y_train[indices]

    # train logistic classifier on generated data
    clf = make_pipeline(StandardScaler(), SGDClassifier(loss='log', max_iter=1000, tol=1e-3))
    clf.fit(X, y)

    # play arg max based on the trained logistic classifier
    commit_plays = clf.predict(X_train[tau:])

    # calculate regret
    arms_played = np.r_[explore_plays, commit_plays]
    regret = np.where(arms_played == y_train, 0, 1)

    return np.cumsum(regret)

def FTL(C, theta_hat, V_t_inv, y_train, S_t, tau, k, Phi, r):
    T, d = C.shape
    FTL_plays = []
    t1, t2, t3 = [[] for i in range(3)] 
    Phi_new = Phi
    for t in range(tau, T):
        if t%500==0: print(t)
        
        start_time = time.time()
        FTL_plays.append( select_best_arm(C[t], theta_hat, k) )
        
        e_a = basis(FTL_plays[-1], k).reshape(k, 1)
        c = C[t].reshape(len(C[t]), 1)
        phi_c_a = ( c @ e_a.T ).flatten()
        phi_c_a = phi_c_a.reshape(len(phi_c_a), 1)
        temp = V_t_inv @ phi_c_a
        mul = 1/(1+ phi_c_a.T @ temp )
        
        V_t_inv = V_t_inv - mul * temp @ (phi_c_a.T @ V_t_inv )
        S_t = S_t + (y_train[t] == FTL_plays[-1]) * phi_c_a
        theta_hat = V_t_inv @ S_t
    
    return FTL_plays

def Follow_The_Leader(X_train, y_train, tau):
    T, d = X_train.shape
    k = len(np.unique(y_train))
    gamma = 0

    # explore to get theta_hat
    explore_plays = np.random.choice(np.arange(k), size=tau)
    Phi = generate_Phi(X_train_pca_norm, explore_plays, k)
    r = np.where(y_train[:tau] == explore_plays, 1, 0).reshape(tau, 1)
    V_t_inv = np.linalg.inv(Phi.T @ Phi + gamma*np.eye(Phi.shape[1]))
    S_t = Phi.T @ r
    theta_hat = V_t_inv @ S_t

    # play follow the leader strategy
    FTL_plays = FTL(X_train_pca_norm, theta_hat, V_t_inv, y_train, S_t, tau, k, Phi, r)

    # calculate regret
    arms_played = np.r_[explore_plays, FTL_plays]
    regret = np.where(arms_played == y_train, 0, 1)

    return np.cumsum(regret)

def UCB_algorithm(X_train, y_train, gamma = 1):
    T, d = X_train.shape
    k = len(np.unique(y_train))
    delta = 1/T
    #gamma = 1
    U = 1
    V_0, S_0 = gamma * np.eye(k*d), np.zeros((k*d, 1))
    log_det_V_0 = np.log(np.linalg.det(V_0))

    I = np.zeros(T)

    V_t, S_t = V_0, S_0
    V_t_inv = np.linalg.inv(V_t)
    for t in range(T):
        if t%400==0: print(t)
        _, log_det_V_t = np.linalg.slogdet(V_t)
        beta_t = np.sqrt(gamma)*U + np.sqrt(2*np.log(1/delta) + log_det_V_t - log_det_V_0 )
        theta_t = V_t_inv @ S_t

        c_t = X_train[t]
        Phi_c_t = np.einsum("i,jk->kij", c_t, np.eye(k))
        Phi_c_t = Phi_c_t.reshape(k, -1)
        i_t = np.argmax( Phi_c_t @ theta_t + beta_t*np.sqrt(np.sum((Phi_c_t @ V_t_inv) * Phi_c_t, axis=1, keepdims=True)) )
        
        # pull arm and observe
        I[t] = i_t

        # update V_t_inv, S_t
        S_t = S_t + (y_train[t] == i_t)*(Phi_c_t[[i_t]].T)
        temp = V_t_inv @ Phi_c_t[[i_t]].T
        mul = 1/(1+ Phi_c_t[[i_t]] @ temp )
        V_t_inv = V_t_inv - mul * temp @ (Phi_c_t[[i_t]] @ V_t_inv )


    regret = np.where(I == y_train, 0, 1)
    return np.cumsum(regret)

def Thompson_sampling(X_train, y_train, gamma=1):
    T, d = X_train.shape
    k = len(np.unique(y_train))
    #gamma = 1
    V_0, S_0 = gamma * np.eye(k*d), np.zeros((k*d, 1))

    Y = np.zeros(T)
    I = np.zeros(T)

    V_t, S_t = V_0, S_0
    V_t_inv = np.linalg.inv(V_t)
    for t in range(T):
        if t%400==0: print(t)
        
        theta_t = V_t_inv @ S_t
        theta_sample = np.random.multivariate_normal(np.squeeze(theta_t), V_t_inv)

        c_t = X_train[t]
        Phi_c_t = np.einsum("i,jk->kij", c_t, np.eye(k))
        Phi_c_t = Phi_c_t.reshape(k, -1)
        i_t = np.argmax(Phi_c_t @ theta_sample)

        # pull arm and observe
        I[t] = i_t

        # update V_t_inv, S_t
        S_t = S_t + (y_train[t] == i_t)*(Phi_c_t[[i_t]].T)
        temp = V_t_inv @ Phi_c_t[[i_t]].T
        mul = 1/(1+ Phi_c_t[[i_t]] @ temp )
        V_t_inv = V_t_inv - mul * temp @ (Phi_c_t[[i_t]] @ V_t_inv )

    regret = np.where(I == y_train, 0, 1)
    return np.cumsum(regret)

def best_linear_model_predictions(X_train, y_train):
    T, d = X_train_pca_norm.shape
    k = len(np.unique(y_train))

    actions_repeat = np.arange(T*10)%10
    Phi_train_full = generate_Phi(np.repeat(X_train, k, axis=0), actions_repeat, k)
    r_train_full = np.where(np.repeat(y_train, k, axis=0) == actions_repeat, 1, 0)

    theta_opt = (np.linalg.inv(Phi_train_full.T @ Phi_train_full) @ Phi_train_full.T) @ r_train_full

    # play arg max based on theta_opt
    optimal_plays = commit(X_train, theta_opt, 0, k)
    
    return optimal_plays


def best_logistic_reg_predictions(X_train, y_train):
    # create optimal logreg model
    clf = make_pipeline(StandardScaler(), SGDClassifier(loss='log', max_iter=1000, tol=1e-3))
    clf.fit(X_train, y_train)

    # play optimally in hindsight based on the trained logistic classifier
    optimal_plays = clf.predict(X_train)
    
    return optimal_plays

if __name__=="__main__":
    T = 50000

    # load downsample, PCA transformed data
    X_train_pca, y_train = load_train_downsampled("train_downsampled_pca_48.csv")
    pca = load_pickle("pca_48.pkl")
    X_train_pca_norm = rescale_norm(X_train_pca)

    X_train_pca_norm, y_train = X_train_pca_norm[:T], y_train[:T]

    n, d = X_train_pca_norm.shape
    k = len(np.unique(y_train))
    
    sim_types = [
        ('ETCW', "Explore then Commit - Model the World"),
        ('ETCB', "Explore then Commit - Model the Bias"),
        ('FTL', "Follow the Leader"),
        ('UCB', "UCB Algorithm"),
        ('TS', "Thompson Sampling"),
    ]

    mean_result, var_result, labels = [[0 for i in range(len(sim_types))] for _ in range(3)]
    for i, (key, label) in enumerate(sim_types):
        # get optimal predictions (in hindsight) for linear and logreg models
        best_linear_predictions = best_linear_model_predictions(X_train_pca_norm, y_train)
        best_logistic_predictions = best_logistic_reg_predictions(X_train_pca_norm, y_train)
        optimal_linear_regret = np.cumsum(np.where(best_linear_predictions == y_train, 0, 1))
        optimal_logistic_regret = np.cumsum(np.where(best_logistic_predictions == y_train, 0, 1))
        
        labels[i] = label
        if key == 'ETCW':
            regret = ETC_world(X_train_pca_norm, y_train)
            mean_result[i] = regret - optimal_linear_regret
        if key == 'ETCB':
            regret = ETC_bias(X_train_pca_norm, y_train)
            mean_result[i] = regret - optimal_logistic_regret
        if key == 'FTL':
            regret = Follow_The_Leader(X_train_pca_norm, y_train, 100)
            mean_result[i] = regret - optimal_linear_regret
        if key == 'UCB':
            regret = UCB_algorithm(X_train_pca_norm, y_train, 0.33)
            mean_result[i] = regret - optimal_linear_regret
        if key == 'TS':
            regret = Thompson_sampling(X_train_pca_norm, y_train, 0.4)
            mean_result[i] = regret - optimal_linear_regret
    
    plot(mean_result, var_result, labels, T)

