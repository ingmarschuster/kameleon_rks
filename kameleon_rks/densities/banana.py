from kameleon_mcmc.distribution.Gaussian import Gaussian
from theano import function
import theano

import kameleon_mcmc.distribution.Banana as B
import numpy as np
import theano.tensor as T


def log_banana_pdf_theano_expr(x, bananicity, V):
    transformed = x.copy()
    transformed = T.set_subtensor(transformed[1], x[1] - bananicity * ((x[0] ** 2) - V))
    transformed = T.set_subtensor(transformed[0], x[0] / T.sqrt(V))
    
    log_determinant_part = 0.
    quadratic_part = -0.5 * transformed.dot(transformed)
    const_part = -0.5 * x.shape[0] * np.log(2 * np.pi)
    
    banana_log_pdf_expr = const_part + log_determinant_part + quadratic_part
    return banana_log_pdf_expr

# build theano functions for log-pdf and gradient
x = T.dvector('x')
bananicity = T.dscalar('bananicity')
V = T.dscalar('V')
banana_log_pdf_expr = log_banana_pdf_theano_expr(x, bananicity, V)
banana_log_pdf_theano = function([x, bananicity, V], banana_log_pdf_expr)
banana_log_pdf_grad_theano = function([x, bananicity, V], theano.gradient.jacobian(banana_log_pdf_expr, x))

def log_banana_pdf(x, bananicity=0.03, V=100, compute_grad=False):
    if not compute_grad:
        return banana_log_pdf_theano(x, bananicity, V)
    else:
        return banana_log_pdf_grad_theano(x, bananicity, V)

def sample_banana(N, D, bananicity=0.03, V=100):
    return  B.Banana(D, bananicity, V).sample(N).samples

def emp_quantiles(X, bananicity=0.03, V=100, quantiles=np.arange(0.1, 1, 0.1)):
    assert(len(X.shape) == 2)
    D = X.shape[1]
    
    substract=bananicity * ((X[:, 0] ** 2) - V)
    divide=np.sqrt(V)
    X[:, 1] -= substract
    X[:, 0] /= divide
    phi = Gaussian(np.zeros(D), np.eye(D))
    quantiles=phi.emp_quantiles(X, quantiles)
    
    # undo changes to X
    X[:, 0] *= divide
    X[:, 1] += substract
    
    return quantiles

def avg_quantile_error(X, bananicity=0.03, V=100, quantiles=np.arange(0.1, 1, 0.1)):
    q = emp_quantiles(X, bananicity, V, quantiles)
    return np.mean(np.abs(q-quantiles))

def norm_of_emp_mean(X):
    return np.linalg.norm(np.mean(X, 0))

class Banana(object):
    def __init__(self, bananicity=0.03, V=100):
        self.bananicity = bananicity
        self.V = V
    
    def log_pdf(self, x):
        return log_banana_pdf(x, self.bananicity, self.V, compute_grad=False)
    
    def grad(self, x):
        return log_banana_pdf(x, self.bananicity, self.V, compute_grad=True)
    
    def emp_quantiles(self, X, quantiles=np.arange(0.1, 1, 0.1)):
        return emp_quantiles(X, self.bananicity, self.V, quantiles)
    
    def set_up(self):
        pass
