from kameleon_rks.gaussian import sample_gaussian, log_gaussian_pdf
from kameleon_rks.running_averages import rank_one_update_mean_covariance_cholesky_lmbda
import numpy as np


class AdaptiveMetropolis():
    """
    Implements the adaptive MH. Performs efficient low-rank updates of Cholesky
    factor of covariance. Covariance itself is not stored/updated, only its Cholesky factor.
    """
    
    def __init__(self, D, nu2, gamma2, schedule=None, acc_star=None):
        """
        D             - Input space dimension
        nu2           - Scaling parameter for covariance
        gamma2        - Exploration parameter. Added to learned variance
        schedule      - Optional. Function that generates adaptation weights
                        given the MCMC iteration number.
                        The weights are used in the stochastic updating of the
                        covariance.
                        
                        If not set, internal covariance is never updated. In that case, call
                        batch_covariance() before using.
        acc_star        Optional: If set, the nu2 parameter is tuned so that
                        average acceptance equals acc_star, using the same schedule
                        as for the chain history update (If schedule is set, otherwise
                        ignored)
        """
        self.D = D
        self.nu2 = nu2
        self.gamma2 = gamma2
        self.schedule = schedule
        self.acc_star = acc_star
        
        # some sanity checks
        assert acc_star is None or acc_star > 0 and acc_star < 1
        if schedule is not None:
            lmbdas = np.array([schedule(t) for t in  np.arange(100)])
            assert np.all(lmbdas > 0)
            assert np.allclose(np.sort(lmbdas)[::-1], lmbdas)
        
        self.initialise()
    
    def initialise(self):
        """
        Initialises internal state. To be called before MCMC chain starts.
        """
        # initialise running averages for covariance
        self.t = 0
        
        if self.schedule is not None:
            # start from scratch
            self.mu = np.zeros(self.D)
            
            # initialise as scaled isotropic, otherwise Cholesky updates fail
            self.L_C = np.eye(self.D) * self.nu2
        else:
            # make user call the set_batch_covariance() function
            self.mu = None
            self.L_C = None     

    def set_batch_covariance(self, Z):
        self.mu = np.mean(Z, axis=0)
        self.L_C = np.linalg.cholesky(self.nu2*np.cov(Z.T)+np.eye(Z.shape[1])*self.gamma2)
    
    def update_scaling(self, accept_prob):
        # generate updating weight
        lmbda = self.schedule(self.t)
        
        # difference desired and actuall acceptance rate
        diff = accept_prob - self.acc_star
            
        self.nu2 = np.exp(np.log(self.nu2) + lmbda * diff)

    def next_iteration(self):
        self.t += 1
        
    def update(self, z_new, previous_accpept_prob):
        """
        Updates the proposal covariance and potentially scaling parameter, according to schedule.
        Note that every call increases a counter that is used for the schedule (if set)
        
        If not schedule is set, this method does not have any effect unless counting.
        
        Parameters:
        z_new                   - A 1-dimensional array of size (D) of.
        previous_accpept_prob   - Acceptance probability of previous iteration
        """
        self.next_iteration()
        
        if self.schedule is not None:
            # generate updating weight
            lmbda = self.schedule(self.t)
            
            # low-rank update of Cholesky, costs O(d^2) only, adding exploration noise on the fly
            self.mu, self.L_C = rank_one_update_mean_covariance_cholesky_lmbda(z_new,
                                                                               lmbda,
                                                                               self.mu,
                                                                               self.L_C,
                                                                               self.nu2,
                                                                               self.gamma2)
            
            # update scalling parameter if wanted
            if self.acc_star is not None:
                self.update_scaling(previous_accpept_prob)
    
    def proposal(self, y):
        """
        Returns a sample from the proposal centred at y, and its log-probability
        """
        if self.schedule is None and (self.mu is None or self.L_C is None):
            raise ValueError("AM has not seen data yet." \
                             "Either call set_batch_covariance() or set update schedule")
        
        # generate proposal
        proposal = sample_gaussian(N=1, mu=y, Sigma=self.L_C, is_cholesky=True)[0]
        proposal_log_prob = log_gaussian_pdf(proposal, y, self.L_C, is_cholesky=True)
        
        # probability of proposing y when would be sitting at proposal is symmetric
        return proposal, proposal_log_prob, proposal_log_prob
    
