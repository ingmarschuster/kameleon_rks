from scipy.misc.common import logsumexp

from kameleon_rks.densities.gaussian import sample_gaussian, log_gaussian_pdf
from kameleon_rks.proposals.ProposalBase import ProposalBase
from kameleon_rks.tools.covariance_updates import log_weights_to_lmbdas, \
    update_mean_cov_L_lmbda
from kameleon_rks.tools.log import Log
import numpy as np


logger = Log.get_logger()

class StaticMetropolis(ProposalBase):
    """
    Implements the classic (isotropic) MH. Allows for tuning the scaling from acceptance rate.
    """
    
    def __init__(self, D, target_log_pdf, step_size, schedule=None, acc_star=None):
        ProposalBase.__init__(self, D, target_log_pdf, step_size, schedule, acc_star)
        
        self.L_C = np.linalg.cholesky(np.eye(D))
    
    def proposal_log_pdf(self, current, proposals):
        log_pdfs = np.zeros(len(proposals))
        for i, proposal in enumerate(proposals):
            log_pdfs[i] = log_gaussian_pdf(proposal, mu=current,
                                           Sigma=self.L_C, is_cholesky=True,
                                           cov_scaling=self.step_size)
        
        return log_pdfs
    
    def proposal(self, current, current_log_pdf, **kwargs):
        if current_log_pdf is None:
            current_log_pdf = self.target_log_pdf(current)
        
        proposal = sample_gaussian(N=1, mu=current, Sigma=self.L_C,
                                   is_cholesky=True, cov_scaling=self.step_size)[0]
        forw_backw_logprob = log_gaussian_pdf(proposal, mu=current,
                                              Sigma=self.L_C, is_cholesky=True, cov_scaling=self.step_size)

        proposal_log_pdf = self.target_log_pdf(proposal)
        
        results_kwargs = {}
        
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
        
        # assume that we have observed fake samples (makes system well-posed)
        # these have covariance gamma2*I, which is a regulariser
        # the mean and log_sum_weights however, is taken from the first set of samples in update
        self.mu = None
        self.L_C = np.eye(D) * np.sqrt(gamma2)
        self.log_sum_weights = None
    
    def set_batch(self, Z):
        # override streaming solution
        self.mu = np.mean(Z, axis=0)
        cov = np.cov(Z.T)
        self.L_C = np.linalg.cholesky(cov + np.eye(self.gamma2))
        self.log_sum_weights = np.log(len(Z))
        
    def update(self, Z, num_new=1, log_weights=None):
        assert(len(Z) >= num_new)
        
        if log_weights is not None:
            assert len(log_weights) == len(Z)
        else:
            log_weights = np.zeros(len(Z))
        
        # nothing observed yet, use average of all observed weights so far
        if self.log_sum_weights is None:
            # this is log mean exp
            self.log_sum_weights = logsumexp(log_weights) - np.log(len(log_weights))
        
        if self.mu is None:
            self.mu = np.mean(Z, axis=0)
        
        # generate lmbdas that correspond to weighted averages
        lmbdas = log_weights_to_lmbdas(self.log_sum_weights, log_weights[-num_new:])
        
        # low-rank update of Cholesky, costs O(d^2) only
        self.mu, self.L_C = update_mean_cov_L_lmbda(Z[-num_new:], self.mu, self.L_C, lmbdas)
        
        # update weights
        stacked = np.hstack((self.log_sum_weights, log_weights[-num_new:]))
        self.log_sum_weights = logsumexp(stacked)


class AdaptiveIndependentMetropolis(AdaptiveMetropolis):
    """
    Implements an independent Gaussian proposal with given parameters.
    
    However, stores mean and covariance in the same fashion as AdaptiveMetropolis
    for debugging purposes, and debug outputs them
    
    Schedule and acc_star are ignored, step size is always 1.
    """
    
    def __init__(self, D, target_log_pdf, gamma2, proposal_mu, proposal_L_C):
        AdaptiveMetropolis.__init__(self, D, target_log_pdf, 1., gamma2)
    
        self.proposal_mu = proposal_mu
        self.proposal_L_C = proposal_L_C
    
    def proposal(self, current, current_log_pdf, **kwargs):
        if current_log_pdf is None:
            current_log_pdf = self.target_log_pdf(current)
        
        proposal = sample_gaussian(N=1, mu=self.proposal_mu, Sigma=self.proposal_L_C,
                                   is_cholesky=True, cov_scaling=self.step_size)[0]
        forw_backw_logprob = log_gaussian_pdf(proposal, mu=self.proposal_mu,
                                              Sigma=self.proposal_L_C, is_cholesky=True,
                                              cov_scaling=self.step_size)
        backw_backw_logprob = log_gaussian_pdf(current, mu=np.zeros(self.D),
                                              Sigma=self.proposal_L_C, is_cholesky=True, cov_scaling=self.step_size)

        proposal_log_pdf = self.target_log_pdf(proposal)
        
        results_kwargs = {}
        
        # probability of proposing current when would be sitting at proposal is symmetric
        return proposal, proposal_log_pdf, current_log_pdf, forw_backw_logprob, backw_backw_logprob, results_kwargs

    def update(self, Z, num_new, log_weights):
        AdaptiveMetropolis.update(self, Z, num_new, log_weights)
        cov = np.dot(self.L_C, self.L_C.T)

        logger.debug("mu: %s" % str(self.mu))
        logger.debug("cov: %s" % str(cov))
