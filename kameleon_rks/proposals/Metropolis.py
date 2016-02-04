from scipy.misc.common import logsumexp

from kameleon_rks.densities.gaussian import sample_gaussian, \
    log_gaussian_pdf_multiple
from kameleon_rks.proposals.ProposalBase import ProposalBase
import kameleon_rks.samplers.tools
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
        log_probs = log_gaussian_pdf_multiple(proposals, mu=current,
                                             Sigma=self.L_C, is_cholesky=True,
                                             cov_scaling=self.step_size)
        return log_probs
    
    def proposal(self, current, current_log_pdf, **kwargs):
        if current_log_pdf is None:
            current_log_pdf = self.target_log_pdf(current)
        
        proposal = sample_gaussian(N=1, mu=current, Sigma=self.L_C,
                                   is_cholesky=True, cov_scaling=self.step_size)[0]
        forw_backw_log_prob = self.proposal_log_pdf(current, proposal[np.newaxis, :])[0]

        proposal_log_pdf = self.target_log_pdf(proposal)
        
        results_kwargs = {}
        
        # probability of proposing current when would be sitting at proposal is symmetric
        return proposal, proposal_log_pdf, current_log_pdf, forw_backw_log_prob, forw_backw_log_prob, results_kwargs

class AdaptiveMetropolis(StaticMetropolis):
    """
    Implements the adaptive MH. Performs efficient low-rank updates of Cholesky
    factor of covariance. Covariance itself is not stored/updated, only its Cholesky factor.
    """
    
    def __init__(self, D, target_log_pdf, step_size, gamma2, schedule=None, acc_star=None):
        StaticMetropolis.__init__(self, D, target_log_pdf, step_size, schedule, acc_star)
        
        self.gamma2 = gamma2
        
        # assume that we have observed fake samples (makes system well-posed)
        # these have covariance gamma2*I, which is a regulariser
        # the mean and log_sum_weights however, is taken from the first set of samples in update
        self.mu = None
        self.L_C = None
        self.log_sum_weights = None
    
    def set_batch(self, Z):
        # override streaming solution
        self.mu = np.mean(Z, axis=0)
        cov = np.cov(Z.T)
        self.L_C = np.linalg.cholesky(cov + np.eye(self.D) * self.gamma2)
        self.log_sum_weights = np.log(len(Z))
        
    def update(self, Z, num_new=1, log_weights=None):
        assert(len(Z) >= num_new)
        
        # dont do anything if no data observed
        if num_new == 0:
            return
        
        if log_weights is not None:
            assert len(log_weights) == len(Z)
        else:
            log_weights = np.zeros(len(Z))

        Z_new = Z[-num_new:]
        log_weights_new = log_weights[-num_new:]
        
        # first update: use first of X and log_weights, and then discard
        if self.log_sum_weights is None:
            # assume have observed fake terms, which is needed for making the system well-posed
            # the L_C says that the fake terms had covariance self.lmbda, which is a regulariser
            self.L_C = np.eye(self.D) * np.sqrt(self.gamma2)
            self.log_sum_weights = log_weights_new[0]
            self.mu = Z_new[0]
            
            Z_new = Z_new[1:]
            log_weights_new = log_weights_new[1:]
            num_new -= 1
            
        # dont do anything if no data observed
        if len(Z_new) == 0:
            return
        
        # generate lmbdas that correspond to weighted averages
        lmbdas = log_weights_to_lmbdas(self.log_sum_weights, log_weights_new)
        
        # low-rank update of Cholesky, costs O(d^2) only
        old_L_C = np.array(self.L_C, copy=True)
        self.mu, self.L_C = update_mean_cov_L_lmbda(Z_new, self.mu, self.L_C, lmbdas)
        
        if np.any(np.isnan(self.L_C)) or np.any(np.isinf(self.L_C)):
            logger.warning("Numerical error while updating Cholesky factor of C.\n"
                           "Before update:\n%s\n"
                           "After update:\n%s\n"
                           "Updating data:\n%s\n"
                           "Updating log weights:\n%s\n"
                           "Updating lmbdas:\n%s\n"
                           
                           % (str(old_L_C), str(self.L_C), str(Z_new), str(log_weights_new), str(lmbdas))
                           )
            raise RuntimeError("Numerical error while updating Cholesky factor of C.")
        
        # update terms and weights
        self.log_sum_weights = logsumexp(list(log_weights) + [self.log_sum_weights])

class AdaptiveIndependentMetropolis(AdaptiveMetropolis):
    """
    Implements an independent Gaussian proposal with given parameters.
    
    However, stores mean and covariance in the same fashion as AdaptiveMetropolis
    for debugging purposes, and debug outputs them
    
    Schedule and acc_star are ignored.
    """
    
    def __init__(self, D, target_log_pdf, step_size, gamma2, proposal_mu, proposal_L_C):
        AdaptiveMetropolis.__init__(self, D, target_log_pdf, step_size, gamma2)
        self.proposal_mu = proposal_mu
        self.proposal_L_C = proposal_L_C
    
        # store all log_weights of all proposals
        self.log_weights = []

    def proposal_log_pdf(self, current, proposals):
        log_probs = log_gaussian_pdf_multiple(proposals, mu=self.proposal_mu,
                                             Sigma=self.proposal_L_C, is_cholesky=True,
                                             cov_scaling=self.step_size)
        return log_probs
    
    def proposal(self, current, current_log_pdf, **kwargs):
        if current_log_pdf is None:
            current_log_pdf = self.target_log_pdf(current)
        
        proposal = sample_gaussian(N=1, mu=self.proposal_mu, Sigma=self.proposal_L_C,
                                   is_cholesky=True, cov_scaling=self.step_size)[0]
        
        forw_backw_log_prob = self.proposal_log_pdf(None, proposal[np.newaxis, :])[0]
        backw_backw_log_prob = self.proposal_log_pdf(None, current[np.newaxis, :])[0]
        

        proposal_log_pdf = self.target_log_pdf(proposal)
        
        results_kwargs = {}
        
        self.log_weights.append(proposal_log_pdf - forw_backw_log_prob)
        
        # probability of proposing current when would be sitting at proposal is symmetric
        return proposal, proposal_log_pdf, current_log_pdf, forw_backw_log_prob, backw_backw_log_prob, results_kwargs
    
    def get_current_ess(self):
        return kameleon_rks.samplers.tools.compute_ess(self.log_weights, normalize=True)

    def update(self, Z, num_new, log_weights):
        AdaptiveMetropolis.update(self, Z, num_new, log_weights)
        cov = np.dot(self.L_C, self.L_C.T)
        var = np.diag(cov)

        logger.debug("mu: %s" % str(self.mu))
        logger.debug("var: %s" % str(var))
        logger.debug("cov: %s" % str(cov))
        logger.debug("norm(mu): %.3f" % np.linalg.norm(self.mu))
        logger.debug("np.mean(var): %.3f" % np.mean(var))
