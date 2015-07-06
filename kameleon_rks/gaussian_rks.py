import numpy as np

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