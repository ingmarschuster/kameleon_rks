from kameleon_rks.densities.gaussian import sample_gaussian, log_gaussian_pdf
from kameleon_rks.proposals.Metropolis import StaticMetropolis
from kameleon_rks.tools.log import Log
from kameleon_rks.tools.running_averages import rank_one_update_mean_covariance_cholesky_lmbda
from kernel_exp_family.examples.tools import visualise_fit
import numpy as np


logger = Log.get_logger()

class StaticLangevin(StaticMetropolis):
    def __init__(self, D, target_log_pdf, grad, step_size, schedule=None, acc_star=None):
        StaticMetropolis.__init__(self, D, target_log_pdf, step_size, schedule, acc_star)
        
        self.grad = grad
    
    def proposal(self, current, current_log_pdf, **kwargs):
        if current_log_pdf is None:
            current_log_pdf = self.target_log_pdf(current)
        
        # potentially save computation via re-using gradient
#         if 'previous_backward_grad' in kwargs:
#             forward_grad = kwargs['previous_backward_grad']
#         else:
        forward_grad = self.grad(current)
        
        # noise covariance square root with step size
        L = np.sqrt(self.step_size) * self.L_C
        
        forward_mu = current + 0.5 * L.dot(L.T.dot(forward_grad))
        proposal = sample_gaussian(N=1, mu=forward_mu, Sigma=L, is_cholesky=True)[0]
        forward_log_prob = log_gaussian_pdf(proposal, forward_mu, L, is_cholesky=True)
        
        backward_grad = self.grad(proposal)
        backward_mu = proposal + 0.5 * L.dot(L.T.dot(backward_grad))
        backward_log_prob = log_gaussian_pdf(proposal, backward_mu, L, is_cholesky=True)
        
        proposal_log_pdf = self.target_log_pdf(proposal)
        
        log_acc_prob = proposal_log_pdf - current_log_pdf + backward_log_prob - forward_log_prob
        log_acc_prob = np.min([0, log_acc_prob])
        
        result_kwargs = {'previous_backward_grad': backward_grad}
        
        return proposal, np.exp(log_acc_prob), proposal_log_pdf, result_kwargs

class AdaptiveLangevin(StaticLangevin):
    def __init__(self, D, target_log_pdf, grad, step_size, gamma2, schedule=None, acc_star=None):
        StaticLangevin.__init__(self, D, target_log_pdf, grad, step_size, schedule, acc_star)

        self.gamma2 = gamma2

        if self.schedule is not None:
            # start from scratch
            self.mu = np.zeros(self.D)
        else:
            # make user call the set_batch function
            self.mu = None
            self.L_C = None

    def set_batch(self, Z):
        # avoid rank-deficient covariances
        if len(Z) > self.D:
            self.mu = np.mean(Z, axis=0)
            self.L_C = np.linalg.cholesky(np.cov(Z.T) + np.eye(Z.shape[1]) * self.gamma2)
    
    def update(self, Z):
        if self.schedule is not None:
            # generate updating weight
            lmbda = self.schedule(self.t)
            
            # low-rank update of Cholesky, costs O(d^2) only, adding exploration noise on the fly
            z_new = Z[-1]
            self.mu, self.L_C = rank_one_update_mean_covariance_cholesky_lmbda(z_new,
                                                                               lmbda,
                                                                               self.mu,
                                                                               self.L_C,
                                                                               self.step_size,
                                                                               self.gamma2)
    
class OracleKernelAdaptiveLangevin(AdaptiveLangevin):
    """
    Implements gradient free kernel adaptive langevin proposal.
    
    Uses the kernel exponential family to estimate the gradient, useing oracle samples.
    """
    
    def __init__(self, D, target_log_pdf, n, surrogate, step_size, gamma2, schedule=None, acc_star=None):
        AdaptiveLangevin.__init__(self, D, target_log_pdf, surrogate.grad, step_size, gamma2, schedule, acc_star)
        
        self.n = n
        self.surrogate = surrogate
    
    def set_batch(self, Z):
        AdaptiveLangevin.set_batch(self, Z)
        
        inds = np.random.permutation(len(Z))[:self.n]
        logger.info("Fitting surrogate gradient model to %d/%d data." % (len(inds), len(Z)))
        self.surrogate.fit(Z[inds])
    
class KernelAdaptiveLangevin(OracleKernelAdaptiveLangevin):
    """
    Implements gradient free kernel adaptive langevin proposal.
    
    Uses the kernel exponential family to estimate the gradient.
    """
    
    def __init__(self, D, target_log_pdf, n, surrogate, step_size, gamma2, schedule=None, acc_star=None):
        OracleKernelAdaptiveLangevin.__init__(self, D, target_log_pdf, n, surrogate, step_size, gamma2, schedule, acc_star)
        
        self.surrogate = surrogate
        self.n = n
    
    def update(self, Z):
        OracleKernelAdaptiveLangevin.update(self, Z)
        
        if self.schedule is not None and len(Z) >= self.n:
            # generate updating probability
            lmbda = self.schedule(self.t)
            
            if np.random.rand() < lmbda:
                # update sub-sample of chain history
                inds = np.random.permutation(len(Z))[:self.n]
                logger.info("Fitting surrogate gradient model to %d/%d data." % (len(inds), len(Z)))
                self.surrogate.fit(Z[inds])
                logger.debug("Updated chain history sub-sample of size %d with probability lmbda=%.3f" % (self.n, lmbda))
