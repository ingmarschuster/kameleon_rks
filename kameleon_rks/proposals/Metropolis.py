from kameleon_rks.proposals.ProposalBase import ProposalBase
from kameleon_rks.densities.gaussian import sample_gaussian
from kameleon_rks.tools.running_averages import rank_one_update_mean_covariance_cholesky_lmbda
import numpy as np


class StaticMetropolis(ProposalBase):
    """
    Implements the classic (isotropic) MH. Allows for tuning the scaling from acceptance rate.
    """
    
    def __init__(self, D, target_log_pdf, step_size, schedule=None, acc_star=None):
        ProposalBase.__init__(self, D, target_log_pdf, step_size, schedule, acc_star)
        
        self.L_C = np.eye(self.D)
    
    def proposal(self, current, current_log_pdf, **kwargs):
        if current_log_pdf is None:
            current_log_pdf = self.target_log_pdf(current)
        
        proposal = sample_gaussian(N=1, mu=current, Sigma=self.L_C * np.sqrt(self.step_size), is_cholesky=True)[0]
        proposal_log_pdf = self.target_log_pdf(proposal)
        
        # probability of proposing current when would be sitting at proposal is symmetric
        acc_log_prob = np.min([0, proposal_log_pdf - current_log_pdf])

        results_kwargs = {}
        
        return proposal, np.exp(acc_log_prob), proposal_log_pdf, results_kwargs

class AdaptiveMetropolis(StaticMetropolis):
    """
    Implements the adaptive MH. Performs efficient low-rank updates of Cholesky
    factor of covariance. Covariance itself is not stored/updated, only its Cholesky factor.
    """
    
    def __init__(self, D, target_log_pdf, step_size, gamma2, schedule=None, acc_star=None):
        StaticMetropolis.__init__(self, D, target_log_pdf, step_size, schedule, acc_star)
        
        self.gamma2 = gamma2
        
        if self.schedule is not None:
            # start from scratch
            self.mu = np.zeros(self.D)
        else:
            # make user call the set_batch function
            self.mu = None
            self.L_C = None

    def set_batch(self, Z):
        self.mu = np.mean(Z, axis=0)
        self.L_C = np.linalg.cholesky(self.step_size * np.cov(Z.T) + np.eye(Z.shape[1]) * self.gamma2)
    
    def update(self, Z):
        if self.schedule is None and self.Z is None:
            raise ValueError("%s has not seen data yet. Call set_batch()" % self.__class__.__name__)
        
        if self.schedule is not None:
            # generate updating weight
            lmbda = self.schedule(self.t)
            
            # low-rank update of Cholesky, costs O(d^2) only, adding exploration noise on the fly
            z_new = Z[-1]
            self.mu, self.L_C = rank_one_update_mean_covariance_cholesky_lmbda(z_new,
                                                                               lmbda,
                                                                               self.mu,
                                                                               self.L_C,
                                                                               1.,
                                                                               self.gamma2)

