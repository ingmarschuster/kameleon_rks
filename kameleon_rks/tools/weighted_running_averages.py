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


def mean_cov_upd_weighted(old_mean, old_cov, old_w_s, samps, weights, logspace = False):
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
    new_cov = ((old_cov + np.outer(old_mean, old_mean)) * old_new_ratio
                        - np.outer(new_mean, new_mean)
                        +  w_samps.T.dot(w_samps))
    return (new_mean, new_cov, new_w_s)


def test():
    s_d = 1.
    D = 2
    num_w1 = 550
    num_w2 = 550
    rvs_w = np.random.randn(D*(num_w1+num_w2)).reshape((num_w1+num_w2, D))*s_d
    w = np.r_[np.ones(num_w1), 2*np.ones(num_w2)]
    assert(len(w) == (num_w1+num_w2))
    rvs = np.r_[rvs_w, rvs_w[-num_w2:]]
    
    rd_idx = np.random.permutation(num_w1+num_w2)
    (rvs_w, w) = (rvs_w[rd_idx], w[rd_idx])
    
    
    
    for logspace in (False, True):
        if logspace:
            weights = log(w)
            w_s = logsumexp(weights)
            abs_toler = 10**(-3)
        else:
            weights = w
            w_s = w.sum()
            abs_toler = 10**(-3)
        mean_true = rvs.mean(0)
        cov_true = np.cov(rvs.T)
        
        (mean_batch, cov_batch, ws_batch) = mean_cov_weighted(rvs_w, weights, logspace)
        assert(np.allclose(mean_true, mean_batch,  atol=abs_toler) and
               np.allclose(cov_true, cov_batch, atol=abs_toler)
               and np.allclose(ws_batch, w_s, atol=abs_toler)
               )
               
        start = 500
        (mean_start, cov_start, ws_start) = mean_cov_weighted(rvs_w[:start], weights[:start],logspace)
        (mean_upd, cov_upd, ws_upd) = mean_cov_upd_weighted(mean_start, cov_start, ws_start, rvs_w[start:], weights[start:], logspace)
        assert(np.allclose(mean_true, mean_upd, atol=abs_toler) and
           np.allclose(cov_true, cov_upd, atol=abs_toler)
           and np.allclose(ws_upd, w_s, atol=abs_toler)
           )
    