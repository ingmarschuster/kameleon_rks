# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 11:52:02 2016

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy as sp
import scipy.stats as stats

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from numpy.linalg import inv




def compute_ess(logweights, normalize = False, ret_logval = False):
    if normalize:
        logweights = logweights - logsumexp(logweights)
    rval = -logsumexp(2*logweights)
    if ret_logval:
        return rval
    else:
        return exp(rval)

def norm_weights(pop):
    prop_w = np.array([s.lweight for s in pop])
    return (prop_w - logsumexp(prop_w))
    
def copy_pop(pop, copy_count):
    return [pop[i] for i in range(len(copy_count)) for _ in range(copy_count[i])]
    
def system_res(pop, weights, resampled_size = None, ess = False, count_only = False):
    if resampled_size is None:
        resampled_size = len(pop)        
    if weights is None:
        prop_w = norm_weights(pop)
    else:
        prop_w = np.array(weights) - logsumexp(weights)
    count = np.uint32(np.zeros(len(pop)))
    Q = np.cumsum(exp(prop_w))
    T = (np.linspace(0,1 - 1 / resampled_size, resampled_size)
         + np.random.rand() / resampled_size)
    i = 0
    j = 0
    while (i < resampled_size and j < len(pop)):
        while Q[j] < T[i]:
            j = j + 1
        count[j] = count[j] + 1
        i = i + 1
    if count_only:
        return count
    else:
        return copy_pop(pop, count)