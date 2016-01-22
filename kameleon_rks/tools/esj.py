# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:33:21 2016

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy as sp
import scipy.stats as stats

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from numpy.linalg import inv




def all_eucl_dist_matr(a, b):
    assert(len(a) == len(b))
    return np.repeat(a, len(a), axis=0) - np.tile(b, (len(a), 1))

def esj(samps, pop_size = 1, mahalanobis = True, outer=False): # expected squared (element-wise) jump
    correction = 1
    if mahalanobis:
        correction = np.linalg.inv(2*np.cov(samps.T))
    dist = np.vstack([all_eucl_dist_matr(samps[start:start+pop_size], samps[start+pop_size: start+2*pop_size]) for start in range(0, len(samps) - 2*pop_size+1, pop_size)])
    if not outer:
        return (dist**2).mean(0).dot(correction)
    else:
        rval = np.zeros([samps.shape[1]]*2)
        for d in dist:
            rval = rval + np.outer(d, d)/ len(dist)
        return rval.dot(correction)