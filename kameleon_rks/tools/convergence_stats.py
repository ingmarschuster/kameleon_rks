from rpy2 import robjects

import numpy as np

def avg_ess(X):
    return np.mean(ess_coda(X))

def min_ess(X):
    return np.min(ess_coda(X))

def ess_coda(X):
        """
        Computes the effective samples size for each column of a 2d-array using R-coda via
        an external R call. The python package rpy2 and the R-library
        "library(coda)" have to be installed. Inspired by Charles Blundell's
        neat little python script :)
        """
        robjects.r('library(coda)')
        D = np.shape(X)[1]
        ESS = np.zeros(D)
        for d in range(D):
            data = X[:,d]
            r_ess = robjects.r['effectiveSize']
            data = robjects.r.matrix(robjects.FloatVector(data), nrow=len(data))
            ESS[d] = r_ess(data)[0]
            
        return ESS

def autocorr(x):
    """
    Computes the ( normalised) auto-correlation function of a
    one dimensional sequence of numbers.
    
    Utilises the numpy correlate function that is based on an efficient
    convolution implementation.
    
    Inputs:
    x - one dimensional numpy array
    
    Outputs:
    Vector of autocorrelation values for a lag from zero to max possible
    """
    
    # normalise, compute norm
    xunbiased = x - np.mean(x)
    xnorm = np.sum(xunbiased ** 2)
    
    # convolve with itself
    acor = np.correlate(xunbiased, xunbiased, mode='same')
    
    # use only second half, normalise
    acor = acor[len(acor) / 2:] / xnorm
    
    return acor

def gelman_rubin(x):
    """ Returns estimate of R for a set of traces.

    The Gelman-Rubin diagnostic tests for lack of convergence by comparing
    the variance between multiple chains to the variance within each chain.
    If convergence has been achieved, the between-chain and within-chain
    variances should be identical. To be most effective in detecting evidence
    for nonconvergence, each chain should have been initialized to starting
    values that are dispersed relative to the target distribution.

    Parameters
    ----------
    x : array-like
      A two-dimensional array containing the parallel traces (minimum 2)
      of some stochastic parameter.

    Returns
    -------
    Rhat : float
      Return the potential scale reduction factor, :math:`\hat{R}`

    Notes
    -----

    The diagnostic is computed by:

      .. math:: \hat{R} = \frac{\hat{V}}{W}

    where :math:`W` is the within-chain variance and :math:`\hat{V}` is
    the posterior variance estimate for the pooled traces. This is the
    potential scale reduction factor, which converges to unity when each
    of the traces is a sample from the target posterior. Values greater
    than one indicate that one or more chains have not yet converged.

    References
    ----------
    Brooks and Gelman (1998)
    Gelman and Rubin (1992)
    
    Copyright
    ---------
    Taken from the pymc package 2.3
    """

    if np.shape(x) < (2,):
        raise ValueError(
            'Gelman-Rubin diagnostic requires multiple chains of the same length.')

    m, N_max = np.shape(x)

    # Calculate between-chain variance
    B_over_n = np.sum((np.mean(x, 1) - np.mean(x)) ** 2) / (m - 1)

    # Calculate within-chain variances
    W = np.sum(
        [(x[i] - xbar) ** 2 for i,
         xbar in enumerate(np.mean(x,
                                   1))]) / (m * (N_max - 1))

    # (over) estimate of variance
    s2 = W * (N_max - 1) / N_max + B_over_n

    # Pooled posterior variance estimate
    V = s2 + B_over_n / m

    # Calculate PSRF
    R = V / W

    return R
