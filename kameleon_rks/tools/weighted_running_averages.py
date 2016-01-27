# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 20:57:37 2016

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy as sp
import scipy.stats as stats

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from numpy.linalg import inv
from choldate._choldate import cholupdate, choldowndate




def mean_cov_weighted(samps, weights, logspace = False):
    if len(samps.shape) > len(weights.shape):
        newsh = list(weights.shape)
        newsh.extend([1]*int(len(samps.shape) - len(weights.shape)))
        weights = weights.reshape(newsh)
    else:
        assert(len(samps.shape) <= len(weights.shape))
    if logspace:
        w_s = logsumexp(weights)
        weights = exp(weights - w_s)
    else:
        w_s = weights.sum()
        weights = weights / w_s
    mu = np.sum(samps*weights,0)
    w_samps = samps * np.sqrt(weights)
    cov = w_samps.T.dot(w_samps) - np.outer(mu, mu)

    return (mu, cov, w_s)        


def mean_cov_upd_weighted(old_mean, old_cov, old_w_s, samps, weights, logspace = False, is_Chol = False):
    """
    Parameters
    ==========
    old_mean   -    old mean
    old_cov    -    old covariance/upper cholesky factor
    old_w_s    -    old sum of weights (in logspace if logspace==True)
    samps      -    new samples
    weights    -    weights of new samples (in logspace if logspace==True)
    logspace   -    weights in logspace if True
    is_Chol    -    old_cov and return value are upper cholesky factors
    
    Returns
    =======
    new_mean   -    new mean
    new_cov    -    new covariance matrix or upper cholesky factor
    """
    if len(samps.shape) > len(weights.shape):
        newsh = list(weights.shape)
        newsh.extend([1]*int(len(samps.shape) - len(weights.shape)))
        weights = weights.reshape(newsh)
    else:
        assert(len(samps.shape) <= len(weights.shape))
    if logspace:
        delta_w_s = logsumexp(weights)
        new_w_s = logsumexp((old_w_s, delta_w_s))
        weights = exp(weights - new_w_s)
        old_new_ratio = exp(old_w_s - new_w_s)
    else:
        delta_w_s = weights.sum()
        new_w_s = old_w_s + delta_w_s
        weights = weights / new_w_s
        old_new_ratio = old_w_s/new_w_s
    
    new_mean = old_mean*old_new_ratio + np.sum(samps*weights,0)
    w_samps = samps * np.sqrt(weights)
    
    if not is_Chol:
        new_cov = ((old_cov + np.outer(old_mean, old_mean)) * old_new_ratio
                            - np.outer(new_mean, new_mean)
                            +  w_samps.T.dot(w_samps))
    else:
        new_cov = old_cov.copy()
        cholupdate(new_cov, old_mean)
        new_cov = new_cov * np.sqrt(old_new_ratio)
        choldowndate(new_cov, new_mean)
        for s in w_samps:
            cholupdate(new_cov, s)
    return (new_mean, new_cov, new_w_s)

def chol_add_diag(L, noise, copy = True):
    D = L.shape[0]
    if copy:
        L = L.copy()
    noise = sqrt(noise)
    e_d = np.zeros(D)
    for d in range(D):
        e_d[d] = noise
        cholupdate(L,  e_d)
        e_d[:] = 0
    return L

def test_weighted_weightedUpdate():
    s_d = 20.
    D = 2
    num_w1 = 550
    num_w2 = 550
    rvs_w = np.random.randn(D*(num_w1+num_w2)).reshape((num_w1+num_w2, D))*s_d
    w = np.r_[np.ones(num_w1), 2*np.ones(num_w2)]
    assert(len(w) == (num_w1+num_w2))
    rvs = np.r_[rvs_w, rvs_w[-num_w2:]]
    mean_true = rvs.mean(0)
    cov_true = np.cov(rvs.T)
    chol_true = sp.linalg.cholesky(cov_true, lower=False)
    
    rd_idx = np.random.permutation(num_w1+num_w2)
    (rvs_w, w) = (rvs_w[rd_idx], w[rd_idx])
    
    
    for chol in (False, True):
        for logspace in (False, True):
            print("Cholesky", chol, "logspace", logspace)
            if logspace:
                weights = log(w)
                w_s = logsumexp(weights)
                abs_toler = 5*10**(-2)
            else:
                weights = w
                w_s = w.sum()
                abs_toler = 5*10**(-2)
            abs_toler = abs_toler * s_d
            

            
            (mean_batch, cov_batch, ws_batch) = mean_cov_weighted(rvs_w, weights, logspace)
            assert(np.allclose(mean_true, mean_batch,  atol=abs_toler) and
                   np.allclose(cov_true, cov_batch, atol=abs_toler)
                   and np.allclose(ws_batch, w_s, atol=abs_toler)
                   )

                   
            start = 500
            (mean_start, cov_start, ws_start) = mean_cov_weighted(rvs_w[:start], weights[:start], logspace)
            if not chol:
                truth = cov_true
            else:
                cov_start = sp.linalg.cholesky(cov_start, lower = False)
                truth = chol_true
            (mean_upd, cov_upd, ws_upd) = mean_cov_upd_weighted(mean_start, cov_start, ws_start, rvs_w[start:], weights[start:], logspace, chol)
            assert(np.allclose(mean_true, mean_upd, atol=abs_toler) and
               np.allclose(truth, cov_upd, atol=abs_toler)
               and np.allclose(ws_upd, w_s, atol=abs_toler)
               )

def test_chol_add_diag():
    D = 4
    cov = -np.ones((D, D)) + np.eye(D) * 5
    L = sp.linalg.cholesky(cov, lower = False)
    
    truth = sp.linalg.cholesky(cov+np.eye(D), lower = False)
    updated = chol_add_diag(L,1)
    if not np.allclose(updated, truth):
        print(updated, 'should have been', truth)
        assert()