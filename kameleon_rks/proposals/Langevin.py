from kameleon_rks.proposals.Metropolis import StaticMetropolis
from kameleon_rks.densities.gaussian import sample_gaussian, log_gaussian_pdf
from kameleon_rks.tools.log import Log
from kameleon_rks.tools.running_averages import rank_one_update_mean_covariance_cholesky_lmbda
import numpy as np


logger = Log.get_logger()

class StaticLangevin(StaticMetropolis):
    def __init__(self, D, target_log_pdf, target_log_grad, step_size, gamma2, eps=1., schedule=None, acc_star=None):
        StaticMetropolis.__init__(self, D, target_log_pdf, step_size, schedule, acc_star)
        
        self.eps = eps
        self.target_log_grad = target_log_grad
    
    def proposal(self, current, current_log_pdf, **kwargs):
        if current_log_pdf is None:
            current_log_pdf = self.target_log_pdf(current)
        
        # potentially save computation via re-using gradient
        if 'previous_backward_grad' in kwargs:
            forward_grad = kwargs['previous_backward_grad']
        else:
            forward_grad = self.target_log_grad(current)
            
        forward_mu = current + self.L_C.dot(self.L_C.T).dot(self.eps * forward_grad)
        proposal = sample_gaussian(N=1, mu=forward_mu, Sigma=self.L_C, is_cholesky=True)[0]
        forward_log_prob = log_gaussian_pdf(proposal, forward_mu, self.L_C, is_cholesky=True)
        
        backward_grad = self.target_log_grad(proposal)
        backward_mu = proposal + self.L_C.dot(self.L_C.T).dot(self.eps * backward_grad)
        backward_log_prob = log_gaussian_pdf(proposal, backward_mu, self.L_C, is_cholesky=True)
        
        proposal_log_pdf = self.target_log_pdf(proposal)
        
        log_acc_prob = proposal_log_pdf - current_log_pdf + backward_log_prob - forward_log_prob
        
        result_kwargs = {'previous_backward_grad': backward_grad}
        
        return proposal, np.exp(log_acc_prob), proposal_log_pdf, result_kwargs

class AdaptiveLangevin(StaticLangevin):
    def __init__(self, D, target_log_pdf, grad, step_size, gamma2, eps=1., schedule=None, acc_star=None):
        StaticLangevin.__init__(self, D, target_log_pdf, grad, step_size, gamma2, eps, schedule, acc_star)

    def set_batch(self, Z):
        self.mu = np.mean(Z, axis=0)
        self.L_C = np.linalg.cholesky(self.setp_size * np.cov(Z.T) + np.eye(Z.shape[1]) * self.gamma2)
    
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
    
class KernelStaticLangevin(StaticLangevin):
    """
    Implements gradient free kernel adaptive langevin proposal.
    
    Uses the kernel exponential family to estimate the gradient.
    """
    
    def __init__(self, D, target_log_pdf, surrogate, step_size, gamma2, eps=1., schedule=None, acc_star=None):
        StaticLangevin.__init__(self, D, target_log_pdf, surrogate.target_log_grad, step_size, gamma2, eps, schedule, acc_star)
        
        self.surrogate = surrogate
    
    def set_batch_covarianc(self, Z):
        StaticLangevin.set_batch(self, Z)
        
        # fit gradient estimator
        self.surrogate.fit(Z)

class KernelAdaptiveLangevin(KernelStaticLangevin):
    """
    Implements gradient free kernel adaptive langevin proposal.
    
    Uses the kernel exponential family to estimate the gradient.
    """
    
    def __init__(self, target_log_pdf, surrogate, step_size, gamma2, eps=1., schedule=None, acc_star=None):
        StaticLangevin.__init__(self, target_log_pdf, surrogate.target_log_grad, step_size, gamma2, eps, schedule, acc_star)
        
        self.surrogate = surrogate
    
    def set_batch(self, Z):
        StaticLangevin.set_batch(self, Z)
        
        # fit gradient estimator
        self.surrogate.fit(Z)

    def update(self, Z):
        if self.schedule is not None:
            # generate updating probability
            lmbda = self.schedule(self.t)
            
            if np.random.rand() < lmbda:
                # update sub-sample of chain history
                self.set_batch(Z[np.random.permutation(len(Z))[:self.n]])
                logger.info("Updated chain history sub-sample of size %d with probability lmbda=%.3f" % (self.n, lmbda))
