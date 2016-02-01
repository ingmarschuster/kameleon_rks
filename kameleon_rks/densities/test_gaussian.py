from numpy.testing.utils import assert_allclose

from kameleon_rks.densities.gaussian import log_gaussian_pdf_multiple, \
    log_gaussian_pdf
import numpy as np


def test_log_gaussian_pdf_multiple_equals_log_gaussian_pdf_looped_full_cov():
    N = 100
    D = 3
    X = np.random.randn(N, D)
    
    mu = np.random.randn(D)
    L_C = np.linalg.cholesky(np.dot(X.T, X) + np.eye(D))
    cov_scaling = 2.
    
    log_pdfs = log_gaussian_pdf_multiple(X, mu, L_C, is_cholesky=True, cov_scaling=cov_scaling)
    grads = log_gaussian_pdf_multiple(X, mu, L_C, is_cholesky=True, compute_grad=True, cov_scaling=cov_scaling)

    print log_pdfs

    for i, x in enumerate(X):
        log_pdf = log_gaussian_pdf(x, mu, L_C, is_cholesky=True, cov_scaling=cov_scaling)
        grad = log_gaussian_pdf(x, mu, L_C, is_cholesky=True, compute_grad=True, cov_scaling=cov_scaling)
        
        assert_allclose(log_pdf, log_pdfs[i])
        assert_allclose(grad, grads[i])
        
def test_log_gaussian_pdf_multiple_equals_log_gaussian_pdf_looped_isotropic_cov():
    N = 100
    D = 3
    X = np.random.randn(N, D)
    
    mu = np.random.randn(D)
    cov_scaling = 2.
    
    log_pdfs = log_gaussian_pdf_multiple(X, mu, cov_scaling=cov_scaling)
    grads = log_gaussian_pdf_multiple(X, mu, compute_grad=True, cov_scaling=cov_scaling)

    print log_pdfs

    for i, x in enumerate(X):
        log_pdf = log_gaussian_pdf(x, mu, cov_scaling=cov_scaling)
        grad = log_gaussian_pdf(x, mu, compute_grad=True, cov_scaling=cov_scaling)
        
        assert_allclose(log_pdf, log_pdfs[i])
        assert_allclose(grad, grads[i])
        