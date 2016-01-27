# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 20:57:37 2016

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

from choldate._choldate import cholupdate, choldowndate
from numpy.testing.utils import assert_allclose
from scipy.misc import logsumexp

import numpy as np
import scipy as sp
from numpy.ma.testutils import assert_close


def mean_cov_weighted(samps, log_weights):
    if len(samps.shape) > len(log_weights.shape):
        newsh = list(log_weights.shape)
        newsh.extend([1] * int(len(samps.shape) - len(log_weights.shape)))
        log_weights = log_weights.reshape(newsh)
    else:
        assert(len(samps.shape) <= len(log_weights.shape))
        
    w_s = logsumexp(log_weights)
    log_weights = np.exp(log_weights - w_s)
    mu = np.sum(samps * log_weights, 0)
    w_samps = samps * np.sqrt(log_weights)
    cov = w_samps.T.dot(w_samps) - np.outer(mu, mu)

    return (mu, cov, w_s)        


def update_mean_cov_weighted(old_mean, old_cov, old_w_s, samps, log_weights):
    """
    Parameters
    ==========
    old_mean    -    old mean
    old_cov     -    old covariance (lower cholesky factor)
    old_w_s     -    old log of sum of weights (logsumexp of log_weights)
    samps       -    new samples
    log_weights -    log_weights of new samples 
    
    Returns
    =======
    new_mean   -    new mean
    new_cov    -    new covariance matrix's lower cholesky factor
    """
    if len(samps.shape) > len(log_weights.shape):
        newsh = list(log_weights.shape)
        newsh.extend([1] * int(len(samps.shape) - len(log_weights.shape)))
        log_weights = log_weights.reshape(newsh)
    else:
        assert(len(samps.shape) <= len(log_weights.shape))
        
    delta_w_s = logsumexp(log_weights)
    new_w_s = logsumexp((old_w_s, delta_w_s))
    log_weights = np.exp(log_weights - new_w_s)
    old_new_ratio = np.exp(old_w_s - new_w_s)
    
    new_mean = old_mean * old_new_ratio + np.sum(samps * log_weights, 0)
    w_samps = samps * np.sqrt(log_weights)
    
    # transpose as choldate expects upper cholesky
    new_cov = old_cov.copy().T
    
    cholupdate(new_cov, old_mean)
    new_cov = new_cov * np.sqrt(old_new_ratio)
    choldowndate(new_cov, new_mean)
    for s in w_samps:
        cholupdate(new_cov, s)
    
    new_cov = new_cov.T
    
    return new_mean, new_cov, new_w_s

def cholesky_update_diag(L, noise, downdate=False):
    
    D = L.shape[0]
    # transpose as choldate expects upper cholesky
    L = L.T
    noise = np.sqrt(noise)
    e_d = np.zeros(D)
    for d in range(D):
        e_d[d] = noise
        
        if downdate:
            choldowndate(L, e_d)
        else:
            cholupdate(L, e_d)
            
        e_d[:] = 0
    # transpose as we return lower cholesky
    return L.T

def test_weighted_weighted_update():
    s_d = 20.
    D = 2
    num_w1 = 550
    num_w2 = 550
    rvs_w = np.random.randn(D * (num_w1 + num_w2)).reshape((num_w1 + num_w2, D)) * s_d
    w = np.r_[np.ones(num_w1), 2 * np.ones(num_w2)]
    assert(len(w) == (num_w1 + num_w2))
    rvs = np.r_[rvs_w, rvs_w[-num_w2:]]
    mean_true = rvs.mean(0)
    cov_true = np.cov(rvs.T)
    chol_true = sp.linalg.cholesky(cov_true, lower=True)
    
    rd_idx = np.random.permutation(num_w1 + num_w2)
    (rvs_w, w) = (rvs_w[rd_idx], w[rd_idx])
    
    
    log_weights = np.log(w)
    w_s = logsumexp(log_weights)
    abs_toler = 5 * 10 ** (-2)
    abs_toler = abs_toler * s_d
    
    (mean_batch, cov_batch, ws_batch) = mean_cov_weighted(rvs_w, log_weights)

    assert_allclose(mean_true, mean_batch, atol=abs_toler)
    assert_allclose(cov_true, cov_batch, atol=abs_toler)
    assert_allclose(ws_batch, w_s, atol=abs_toler)

           
    start = 500
    (mean_start, cov_start, ws_start) = mean_cov_weighted(rvs_w[:start], log_weights[:start])
    cov_start = sp.linalg.cholesky(cov_start, lower=True)
    truth = chol_true
    
    (mean_upd, cov_upd, ws_upd) = update_mean_cov_weighted(mean_start, cov_start, ws_start, rvs_w[start:], log_weights[start:])
    
    assert_allclose(mean_true, mean_upd, atol=abs_toler)
    assert_allclose(truth, cov_upd, atol=abs_toler)
    assert_allclose(ws_upd, w_s, atol=abs_toler)

def test_cholesky_update_diag():
    D = 4
    cov = -np.ones((D, D)) + np.eye(D) * 5
    L = sp.linalg.cholesky(cov, lower=True)
    
    noise = 2
    truth = sp.linalg.cholesky(cov + np.eye(D) * noise, lower=True)
    updated = cholesky_update_diag(L, noise)
    assert_allclose(updated, truth)

def test_cholesky_update_diag_downdate():
    D = 4
    cov = -np.ones((D, D)) + np.eye(D) * 5
    
    noise = 2
    L = sp.linalg.cholesky(cov + np.eye(D) * noise, lower=True)
    updated = cholesky_update_diag(L, noise, downdate=True)
    
    truth = sp.linalg.cholesky(cov, lower=True)
    assert_allclose(updated, truth)

def test_covariance_updates():
    D = 2
    Z = np.random.randn(100, D)
    Z2 = np.random.randn(2 * len(Z), D)

    full_cov = np.cov(np.vstack((Z, Z2)).T)
    full_mean = np.mean(np.vstack((Z, Z2)), 0)
    full_cov_L = np.linalg.cholesky(full_cov)
    
    runnung_cov_L = np.linalg.cholesky(np.cov(Z.T))
    running_mean = np.mean(Z, 0) 
    running_weight_sum = np.log(np.sum(np.ones(len(Z))))
    
    for i in range(int(len(Z2) / 2)):
        log_weights = np.zeros(2)
        samples = Z2[i:(i + 1)]
        running_mean, runnung_cov_L, running_weight_sum = update_mean_cov_weighted(running_mean,
                                                                                 runnung_cov_L,
                                                                                 running_weight_sum,
                                                                                 samples,
                                                                                 log_weights)
    
    assert_allclose(full_mean, full_mean)
    assert_allclose(runnung_cov_L, full_cov_L)
    assert_close(running_weight_sum, np.log(len(Z) + len(Z2)))
