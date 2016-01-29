from kameleon_rks.densities.gaussian import sample_gaussian, log_gaussian_pdf
from kameleon_rks.proposals.Metropolis import StaticMetropolis,\
    AdaptiveMetropolis
from kameleon_rks.tools.log import Log
import numpy as np


logger = Log.get_logger()

class StaticLangevin(StaticMetropolis):
    def __init__(self, D, target_log_pdf, grad, step_size, schedule=None, acc_star=None):
        StaticMetropolis.__init__(self, D, target_log_pdf, step_size, schedule, acc_star)
        
        self.grad = grad
        self.manual_gradient_step_size = None
    
    def proposal(self, current, current_log_pdf, **kwargs):
        if current_log_pdf is None:
            current_log_pdf = self.target_log_pdf(current)
        
        # potentially save computation via re-using gradient
#         if 'previous_backward_grad' in kwargs:
#             forward_grad = kwargs['previous_backward_grad']
#         else:
        forward_grad = self.grad(current)
        
        gradient_step_size = self.manual_gradient_step_size if self.manual_gradient_step_size is not None else self.step_size
        
        forward_mu = current + 0.5 * gradient_step_size * self.L_C.dot(self.L_C.T.dot(forward_grad))
        proposal = sample_gaussian(N=1, mu=forward_mu, Sigma=self.L_C,
                                   is_cholesky=True, cov_scaling=self.step_size)[0]
        forward_log_prob = log_gaussian_pdf(proposal, forward_mu, self.L_C,
                                            is_cholesky=True,
                                            cov_scaling=self.step_size)
        
        backward_grad = self.grad(proposal)
        backward_mu = proposal + 0.5 * gradient_step_size * self.L_C.dot(self.L_C.T.dot(backward_grad))
        try:
            backward_log_prob = log_gaussian_pdf(proposal, backward_mu, self.L_C,
                                                is_cholesky=True,
                                                cov_scaling=self.step_size)
        except Exception:
            pass
        
        proposal_log_pdf = self.target_log_pdf(proposal)
        
        result_kwargs = {'previous_backward_grad': backward_grad}
        
        return proposal, proposal_log_pdf, current_log_pdf, forward_log_prob, backward_log_prob, result_kwargs

class AdaptiveLangevin(StaticLangevin, AdaptiveMetropolis):
    def __init__(self, D, target_log_pdf, grad, step_size, schedule=None, acc_star=None):
        StaticLangevin.__init__(self, D, target_log_pdf, grad, step_size, schedule, acc_star)
        
        # gamma2 in AM is not used
        gamma2_dummy = 0.1
        AdaptiveMetropolis.__init__(self, D, target_log_pdf, step_size, gamma2_dummy, schedule, acc_star)
        
    def proposal(self, current, current_log_pdf, **kwargs):
        return StaticLangevin.proposal(self, current, current_log_pdf)

    def update(self, Z, num_new=1, log_weights=None):
        return AdaptiveMetropolis.update(self, Z, num_new, log_weights)
    
    def set_batch(self, Z, log_weights=None):
        return AdaptiveMetropolis.set_batch(self, Z, log_weights)
    
class OracleKernelAdaptiveLangevin(AdaptiveLangevin):
    """
    Implements gradient free kernel adaptive langevin proposal.
    
    Uses the kernel exponential family to estimate the gradient, useing oracle samples.
    """
    
    def __init__(self, D, target_log_pdf, surrogate, step_size, schedule=None, acc_star=None):
        AdaptiveLangevin.__init__(self, D, target_log_pdf, surrogate.grad, step_size, schedule, acc_star)
        
        self.surrogate = surrogate
        
        assert surrogate.supports_weights()
        assert surrogate.supports_update_fit()
        
    
    def set_batch(self, Z, log_weights=None):
        AdaptiveLangevin.set_batch(self, Z, log_weights)
        
        if log_weights is None:
            weights = np.ones(len(Z))
        else:
            weights = np.exp(log_weights)
        
        logger.debug("Fitting surrogate gradient model to %d data." % len(Z))
        self.surrogate.fit(Z, weights)
    
class KernelAdaptiveLangevin(OracleKernelAdaptiveLangevin):
    """
    Implements gradient free kernel adaptive langevin proposal.
    
    Uses the kernel exponential family to estimate the gradient.
    """
    
    def __init__(self, D, target_log_pdf, surrogate, step_size, schedule=None, acc_star=None):
        OracleKernelAdaptiveLangevin.__init__(self, D, target_log_pdf, surrogate, step_size, schedule, acc_star)
        
    def update(self, Z, num_new=1, log_weights=None):
        OracleKernelAdaptiveLangevin.update(self, Z, num_new, log_weights)
        
        if log_weights is None:
            weights = np.ones(len(Z))
        else:
            weights = np.exp(log_weights)
        
        # update surrogate
        logger.debug("Updating surrogate gradient model using %d data." % num_new)
        self.surrogate.update_fit(Z[-num_new:], weights[-num_new:])
