from scipy.misc.common import logsumexp

from kameleon_rks.densities.gaussian import sample_gaussian, log_gaussian_pdf
from kameleon_rks.proposals.ProposalBase import ProposalBase
from kameleon_rks.tools.covariance_updates import cholupdate_diag, \
    log_weights_to_lmbdas, update_mean_cov_L_lmbda
import numpy as np


class StaticMetropolis(ProposalBase):
    """
    Implements the classic (isotropic) MH. Allows for tuning the scaling from acceptance rate.
    """
    
    def __init__(self, D, target_log_pdf, step_size, schedule=None, acc_star=None):
        ProposalBase.__init__(self, D, target_log_pdf, step_size, schedule, acc_star)
        
        self.L_C = np.linalg.cholesky(np.eye(D))
    
    def proposal(self, current, current_log_pdf, **kwargs):
        if current_log_pdf is None:
            current_log_pdf = self.target_log_pdf(current)
        
        # scale and add noise, O(D^2) + O(D)
        self.L_C *= np.sqrt(self.step_size)
        self.L_C = cholupdate_diag(self.L_C, self.gamma2)
        
        # O(D^2)
        proposal = sample_gaussian(N=1, mu=current, Sigma=self.L_C, is_cholesky=True)[0]
        forw_backw_logprob = log_gaussian_pdf(proposal, mu=current, Sigma=self.L_C, is_cholesky=True)
        proposal_log_pdf = self.target_log_pdf(proposal)
        
        results_kwargs = {}
        
        # remove noise and unscale, O(D^2) + O(D)
        self.L_C = cholupdate_diag(self.L_C, self.gamma2, downdate=True)
        self.L_C /= np.sqrt(self.step_size)
        
        # probability of proposing current when would be sitting at proposal is symmetric
        return proposal, proposal_log_pdf, current_log_pdf, forw_backw_logprob, forw_backw_logprob, results_kwargs

class AdaptiveMetropolis(StaticMetropolis):
    """
    Implements the adaptive MH. Performs efficient low-rank updates of Cholesky
    factor of covariance. Covariance itself is not stored/updated, only its Cholesky factor.
    """
    
    def __init__(self, D, target_log_pdf, step_size, gamma2, schedule=None, acc_star=None):
        StaticMetropolis.__init__(self, D, target_log_pdf, step_size, schedule, acc_star)
        
        assert gamma2 > 0 and gamma2 < 1
        
        self.gamma2 = gamma2
        
        self.mu = np.zeros(D)
        
        # assume that we have observed D samples so far
        self.log_sum_weights = np.log(D)

    def set_batch(self, Z, log_weights=None):
        if log_weights is None:
            weights = np.ones(len(Z))
        else:
            weights = np.exp(log_weights)
        
        self.mu = np.average(Z, axis=0, aweights=weights)
        self.L_C = np.linalg.cholesky(self.step_size * np.cov(Z.T, aweights=weights) + np.eye(self.D) * self.gamma2)
        self.log_sum_weights = logsumexp(log_weights)
        
    def update(self, Z, num_new=1, log_weights=None):
        assert(len(Z) >= num_new)
        
        if log_weights is not None:
            assert len(log_weights) == len(Z)
        else:
            log_weights = np.zeros(len(Z))
        
        # generate lmbdas that correspond to weighted averages
        lmbdas = log_weights_to_lmbdas(self.log_sum_weights, log_weights[-num_new:])
        
        # low-rank update of Cholesky, costs O(d^2) only
        self.mu, self.L_C = update_mean_cov_L_lmbda(Z[-num_new:], self.mu, self.L_C, lmbdas)
        
        # update weights
        self.log_sum_weights = logsumexp([self.log_sum_weights, log_weights[-num_new:]])
