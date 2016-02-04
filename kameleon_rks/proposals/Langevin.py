from abc import abstractmethod

from kameleon_rks.densities.gaussian import sample_gaussian, \
    log_gaussian_pdf_multiple
from kameleon_rks.proposals.Metropolis import StaticMetropolis, \
    AdaptiveMetropolis
from kameleon_rks.tools.log import Log
import numpy as np


logger = Log.get_logger()

class StaticLangevin(StaticMetropolis):
    def __init__(self, D, target_log_pdf, grad, step_size, schedule=None, acc_star=None):
        StaticMetropolis.__init__(self, D, target_log_pdf, step_size, schedule, acc_star)
        
        self.grad = grad
        
        # members hidden from constructor
        self.manual_gradient_step_size = None
        self.do_preconditioning = False
        
        self.forward_drift_norms = []
    
    def _compute_drift(self, current):
        grad = self.grad(current)
        
        gradient_step_size = self.manual_gradient_step_size if self.manual_gradient_step_size is not None else self.step_size
        
        if self.do_preconditioning:
            drift = 0.5 * gradient_step_size * self.L_C.dot(self.L_C.T.dot(grad))
        else:
            drift = 0.5 * gradient_step_size * grad
        
        return drift
    
    def proposal_log_pdf(self, current, proposals, drift=None):
        if drift is None:
            drift = self._compute_drift(current)
        
        mu = current + drift
        
        log_probs = log_gaussian_pdf_multiple(proposals, mu=mu, Sigma=self.L_C,
                                              is_cholesky=True,
                                              cov_scaling=self.step_size)
        
        return log_probs
        
    
    def proposal(self, current, current_log_pdf, **kwargs):
        if current_log_pdf is None:
            current_log_pdf = self.target_log_pdf(current)
        
        forward_drift = self._compute_drift(current)
        forward_mu = current + forward_drift

        forward_drift_norm = np.linalg.norm(forward_drift)
        logger.debug("Norm of forward drift: %.3f" % forward_drift_norm)
        self.forward_drift_norms += [forward_drift_norm]
        
        proposal = sample_gaussian(N=1, mu=forward_mu, Sigma=self.L_C,
                                   is_cholesky=True, cov_scaling=self.step_size)[0]
        forward_log_prob = self.proposal_log_pdf(current, proposal[np.newaxis, :],
                                                 drift=forward_drift)[0]
        
        backward_drift = self._compute_drift(proposal)
        backward_log_prob = self.proposal_log_pdf(proposal, current[np.newaxis, :],
                                         drift=backward_drift)[0]
            
        proposal_log_pdf = self.target_log_pdf(proposal)
        
        result_kwargs = {}
        
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
    
    def set_batch(self, Z):
        return AdaptiveMetropolis.set_batch(self, Z)
    
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
    
    def set_batch(self, Z):
        AdaptiveLangevin.set_batch(self, Z)
        
        logger.debug("Fitting surrogate gradient model to %d data." % len(Z))
        self.surrogate.fit(Z)
    
    @abstractmethod
    def get_parameters(self):
        d = AdaptiveLangevin.get_parameters(self)
        mine = {'lmbda': self.surrogate.lmbda, 'sigma': self.surrogate.sigma}
        d.update(mine)
        return d
    
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
            log_weights = np.log(np.ones(len(Z)))

        # update surrogate
        logger.debug("Updating surrogate gradient model using %d data." % num_new)
        self.surrogate.update_fit(Z[-num_new:], log_weights[-num_new:])
