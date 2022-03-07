import numpy as np
import matplotlib.pyplot as plt
from g_optimal_design import allocate_frank_wolfe

def homework_plot(X, f_star, f_hat, show=True):
    if not show: return
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # plot 1
    ax1.hist(X)

    # plot 2
    ax2.set_yscale("log")
    ax2.plot(e)

    # plot 3
    ax3.plot(X, f_star)
    ax3.plot(X, f_hat)

    plt.show()

n =300
X = np.concatenate( ( np . linspace (0 ,1 ,50) , 0.25+ 0.01* np . random . randn (250) ) , 0)
X = np.sort( X )

K = np.zeros (( n , n ) )
for i in range( n ):
    for j in range( n ) :
        K[i , j] = 1+ min( X[i], X[j])
e , v = np . linalg . eigh ( K ) # eigenvalues are increasing in order
d = 30
Phi = np.real(v @ np.diag(np.sqrt(np.abs(e))) )[:,(n-d)::]

def f ( x ) :
    return -x **2 + x*np.cos(8*x) + np.sin(15*x)

f_star = f ( X )

theta = np.linalg.lstsq( Phi, f_star, rcond = None )[0]
f_hat = Phi @ theta

homework_plot(X, f_star, f_hat, show=False)

def observe ( idx ) :
    return f( X[idx]) + np.random.randn(len(idx))

def sample_and_estimate(X, lbda, tau):
    n, d = X.shape
    reg = 1e-6 # we can add a bit of regulari zation to avoid divide by 0
    idx = np.random.choice(np.arange(n), size=tau, p=lbda)
    y = observe(idx)

    XtX = X [ idx ]. T @ X [ idx ]
    XtY = X [ idx ]. T @ y
    
    theta = np.linalg.lstsq( XtX + reg*np.eye(d), XtY, rcond = None )[0]
    return Phi @ theta, XtX

def plot_func_estimate_1():
    p = 1. * np.arange(len(X)) / (len(X) - 1)
    plt.plot(X, p, color='tab:blue', label=r'CDF of uniform distribution over $\chi$')

    plt.plot(X, np.cumsum(lbda_G), color='tab:orange', label='CDF of G-optimal allocation')

    plt.legend(loc='upper left')
    plt.xlabel('x')
    plt.ylabel(r'CDF - P(X $\leq$ x)')
    plt.title(r'CDF of a distribution uniform over $\chi$ vs CDF of the G-optimal allocation,' + \
        '\nThe G-optimal allocation explores points away from 0.25 as they have higher uncertainty')
    plt.show()
    pass

def plot_func_estimate_2():
    plt.plot(X, f_star, color='tab:blue', label=r'f_star - true f')
    plt.plot(X, f_G_Phi, color='tab:orange', linestyle='dashed', label=r'f_G_Phi - f estimated with G optimal exploration')
    plt.plot(X, f_unif_Phi, color='tab:red', label='f_unif_Phi - f estimated with uniform exploration')

    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel(r'value of f')
    plt.title(r'f_star vs f_G_Phi vs f_unif_Phi')
    plt.show()
    pass

def plot_func_estimate_3():
    plt.plot(X, np.abs(f_G_Phi - f_star), color='tab:blue', label=r'|f_G_Phi - f_star|')
    plt.plot(X, np.abs(f_unif_Phi - f_star), color='tab:orange', label=r'|f_unif_Phi - f_star|')
    plt.plot(X, np.sqrt(d/n)*np.ones_like(X), color='tab:grey', linestyle='dashed', label=r'$\sqrt{d/n}$')
    plt.plot(X, conf_G, color='tab:blue', marker='o', label='conf_G')
    plt.plot(X, conf_unif, color='tab:orange', marker='o', label='conf_unif')

    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.title(r'absolute devitation of the f estimate vs true for G-optimal and uniform exploration design')
    plt.show()
    pass


T = 1000

_, allocation, lbda_G = allocate_frank_wolfe( Phi, T )
f_G_Phi , A = sample_and_estimate(Phi, lbda_G, T)
conf_G = np.sqrt(np.sum(Phi @ np.linalg.inv(A) * Phi, axis =1) )

lbda_U = np.ones(n)/n
f_unif_Phi, A = sample_and_estimate(Phi, lbda_U, T)
conf_unif = np.sqrt(np.sum(Phi @ np.linalg.inv(A) * Phi , axis =1) )

plot_func_estimate_1()
plot_func_estimate_2()
plot_func_estimate_3()