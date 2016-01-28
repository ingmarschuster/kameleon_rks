from choldate._choldate import cholupdate, choldowndate
from scipy.misc.common import logsumexp

import numpy as np


def update_mean_lmbda(X, old_mean, lmbdas):
    assert len(X) == len(lmbdas)
    
    mean = old_mean
    for x, lmbda in zip(X, lmbdas):
        mean = (1 - lmbda) * mean + lmbda * x
    
    return mean

def update_mean_cov_L_lmbda(X, old_mean, old_cov_L, lmbdas):
    assert len(X) == len(lmbdas)
    
    # work on upper triangular cholesky internally
    old_cov_R = old_cov_L.T
    
    mean = old_mean
    for x, lmbda in zip(X, lmbdas):
        old_cov_R *= np.sqrt(1 - lmbda)
        update_vec = np.sqrt(lmbda) * (x - mean)
        cholupdate(old_cov_R, update_vec)
        mean = (1 - lmbda) * mean + lmbda * x
    
    # transform back to lower triangular version
    cov_L = old_cov_R.T
    
    return mean, cov_L

def weights_to_lmbdas(sum_old_weights, new_weights):
    N = len(new_weights)
    lmbdas = np.zeros(N)
    
    for i, new_weight in enumerate(new_weights):
        sum_old_weights += new_weight
        lmbdas[i] = new_weight / (sum_old_weights)
    
    return lmbdas

def log_weights_to_lmbdas(log_sum_old_weights, log_new_weights):
    N = len(log_new_weights)
    lmbdas = np.zeros(N)
    
    for i, log_new_weight in enumerate(log_new_weights):
        log_sum_old_weights = logsumexp([log_sum_old_weights, log_new_weight])
        log_lmbda = log_new_weight - log_sum_old_weights
        lmbdas[i] = np.exp(log_lmbda)
    
    return lmbdas

def cholupdate_diag(L, noise, downdate=False):
    # check: can this be a lower cholesky?
    assert(L[0, -1] == 0)
    D = L.shape[0]
    
    noise_sqrt = np.sqrt(noise)
    
    # work on upper triangular cholesky internally
    L = L.T
    
    e_d = np.zeros(D)
    for d in range(D):
        e_d[:] = 0
        e_d[d] = noise_sqrt
        
        # That is O(D^2) and therefore not efficient when used in a loop
        if downdate:
            choldowndate(L, e_d)
        else:
            cholupdate(L, e_d)
        
        # TODO:
        # in contrast, can do a simplified update when knowing that e_d is sparse
        # manual Cholesky update (only doing the d-th component of algorithm on
        # https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update
#             # wiki (MB) code:
#             r = sqrt(L(k,k)^2 + x(k)^2);
#             c = r / L(k, k);
#             s = x(k) / L(k, k);
#             L(k, k) = r;
#             L(k+1:n,k) = (L(k+1:n,k) + s*x(k+1:n)) / c;
#             x(k+1:n) = c*x(k+1:n) - s*L(k+1:n,k);
        
    # transform back to lower triangular version
    L = L.T
    
    return L