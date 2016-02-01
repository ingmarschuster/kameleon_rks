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
    
    def proposal(self, current, current_log_pdf, **kwargs):
        if current_log_pdf is None:
            current_log_pdf = self.target_log_pdf(current)
        
        proposal = sample_gaussian(N=1, mu=current, Sigma=self.L_C,
                                   is_cholesky=True, cov_scaling=self.step_size)[0]
        try:
            forw_backw_logprob = log_gaussian_pdf(proposal, mu=current,
                                                  Sigma=self.L_C, is_cholesky=True, cov_scaling=self.step_size)
        except Exception as e:
            logger.error("Could not compute forward probability.")
            logger.error("current:", current)
            logger.error("proposal:", proposal)
            logger.error("L_C:", self.L_C)
            logger.error("mu:", self.mu)
            
            raise e

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
        
        # assume that we have observed D samples so far (makes system well-posed)
        # these have zero mean and covariance gamma2*I
        self.mu = np.zeros(D)
        self.L_C = np.eye(D) * np.sqrt(gamma2)
        self.log_sum_weights = np.log(D)
    
    def proposal(self, current, current_log_pdf, **kwargs):
        # mixture proposal with isotropic random walk
        if np.random.rand() < self.gamma2:
            use_adaptive_proposal = False
        else:
            use_adaptive_proposal = True
        
        if use_adaptive_proposal:
            logger.debug("Proposal with learned covariance")
            return StaticMetropolis.proposal(self, current, current_log_pdf, **kwargs)
        else:
            if current_log_pdf is None:
                current_log_pdf = self.target_log_pdf(current)
            
            logger.debug("Proposal with isotropic covariance")
            proposal = sample_gaussian(N=1, mu=current, cov_scaling=self.step_size)[0]
            forw_backw_logprob = log_gaussian_pdf(proposal, mu=current, cov_scaling=self.step_size)
            
            proposal_log_pdf = self.target_log_pdf(proposal)
            results_kwargs = {}
                
            # probability of proposing current when would be sitting at proposal is symmetric
            return proposal, proposal_log_pdf, current_log_pdf, forw_backw_logprob, forw_backw_logprob, results_kwargs

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
        
        # generate lmbdas that correspond to weighted averages
        lmbdas = log_weights_to_lmbdas(self.log_sum_weights, log_weights[-num_new:])
        
        # low-rank update of Cholesky, costs O(d^2) only
        self.mu, self.L_C = update_mean_cov_L_lmbda(Z[-num_new:], self.mu, self.L_C, lmbdas)
        
        # update weights
        stacked = np.hstack((self.log_sum_weights, log_weights[-num_new:]))
        self.log_sum_weights = logsumexp(stacked)
