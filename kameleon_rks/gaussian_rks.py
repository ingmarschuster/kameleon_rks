from scipy.spatial.distance import squareform, pdist

import numpy as np


def gamma_median_heuristic(Z, num_subsample=1000):
    """
    Computes the median pairwise distance in a random sub-sample of Z.
    Returns a \gamma for k(x,y)=\exp(-\gamma ||x-y||^2), according to the median heuristc,
    i.e. it corresponds to \sigma in k(x,y)=\exp(-0.5*||x-y||^2 / \sigma^2) where
    \sigma is the median distance. \gamma = 0.5*(\sigma^2)
    """
    inds = np.random.permutation(len(Z))[:np.max([num_subsample, len(Z)])]
    dists = squareform(pdist(Z[inds], 'sqeuclidean'))
    median_dist = np.median(dists[dists > 0])
    sigma = np.sqrt(0.5 * median_dist)
    gamma = 0.5 * (sigma ** 2)
    
    return gamma

def sample_basis(D, m, gamma):
    omega = gamma * np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    return omega, u

def feature_map_single(x, omega, u):
    m = 1 if np.isscalar(u) else len(u)
    return np.cos(np.dot(x, omega) + u) * np.sqrt(2. / m)

def feature_map(X, omega, u):
    m = 1 if np.isscalar(u) else len(u)
    
    projection = np.dot(X, omega) + u
    np.cos(projection, projection)
    projection *= np.sqrt(2. / m)
    return projection

def feature_map_derivative_d(X, omega, u, d):
    m = 1 if np.isscalar(u) else len(u)
    
    projection = np.dot(X, omega) + u
    np.sin(projection, projection)
        
    projection *= omega[d, :]
    projection *= np.sqrt(2. / m)
    return -projection

def feature_map_derivative2_d(X, omega, u, d):
    Phi2 = feature_map(X, omega, u)
    Phi2 *= omega[d, :] ** 2
    
    return -Phi2

def feature_map_grad_single(x, omega, u):
    D, m = omega.shape
    grad = np.zeros((D, m))
    
    for d in range(D):
        grad[d, :] = feature_map_derivative_d(x, omega, u, d)
    
    return grad

def feature_map_derivatives_loop(X, omega, u):
    m = 1 if np.isscalar(u) else len(u)
    N = X.shape[0]
    D = X.shape[1]
    
    projections = np.zeros((D, N, m))
    projection = np.dot(X, omega) + u
    np.sin(projection, projection)
    for d in range(D):
        projections[d, :, :] = projection
        projections[d, :, :] *= omega[d, :]
    
    projections *= -np.sqrt(2. / m)
    return projections

def feature_map_derivatives2_loop(X, omega, u):
    m = 1 if np.isscalar(u) else len(u)
    N = X.shape[0]
    D = X.shape[1]
    
    projections = np.zeros((D, N, m))
    Phi2 = feature_map(X, omega, u)
    for d in range(D):
        projections[d, :, :] = -Phi2
        projections[d, :, :] *= omega[d, :] ** 2
        
    return projections

def feature_map_derivatives(X, omega, u):
    return feature_map_derivatives_loop(X, omega, u)

def feature_map_derivatives2(X, omega, u):
    return feature_map_derivatives2_loop(X, omega, u)
