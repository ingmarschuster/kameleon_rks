from kameleon_rks.tools.running_averages import rank_one_update_mean_covariance_cholesky_lmbda
import numpy as np


if __name__ == '__main__':
    N = 10000
    D = 2
    gamma2 = 1.0
    x_std = 1.
    
    
    X = np.random.randn(N, D) * x_std
    
    print "batch"
    print "mean:", np.mean(X, 0)
    print "cov:\n", np.cov(X.T) + np.eye(D) * gamma2
    
    lmbda = 0.001
    nu2 = 1.
    cov_L = np.eye(D)
    mean = np.zeros(D)
    for i in range(N):
        u = np.random.randn(D) * x_std
        mean, cov_L = rank_one_update_mean_covariance_cholesky_lmbda(u, lmbda, mean, cov_L, nu2, gamma2)
    
    print "rank one updates"
    print "mean:", mean
    print "cov:\n", np.dot(cov_L, cov_L.T)
