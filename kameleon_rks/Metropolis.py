import numpy as np


class Metropolis():
    """
    Implements the classic (isotropic) MH. Allows for tuning the scaling from acceptance rate though.
    """
    
    def __init__(self, D, nu2, schedule=None, acc_star=None):
        """
        D             - Input space dimension
        nu2           - Scaling parameter for covariance
        schedule      - Optional. Function that generates adaptation weights
                        given the MCMC iteration number.
                        The weights are used in the updating of the scaling, if acc_star is set.
        acc_star        Optional: If set, the nu2 parameter is tuned so that
                        average acceptance equals acc_star (If schedule is set, otherwise
                        ignored)
        """
        self.D = D
        self.nu2 = nu2
        self.schedule = schedule
        self.acc_star = acc_star
        
        # some sanity checks
        assert acc_star > 0 and acc_star < 1
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
    
    def set_batch_covariance(self, Z):
        pass
    
    def update_scaling(self, accept_prob):
        assert(self.schedule is not None)
        self.nu2 = np.exp(np.log(self.nu2) + self.schedule(self.t) * (accept_prob - self.acc_star))

    def next_iteration(self):
        self.t += 1
        
    def update(self, z_new, previous_accpept_prob):
        """
        Updates the proposal scaling parameter, according to schedule.
        Note that every call increases a counter that is used for the schedule (if set)
        
        If not schedule is set, this method does not have any effect.
        
        Parameters:
        z_new                   - A 1-dimensional array of size (D) of. Ignored
        previous_accpept_prob   - Acceptance probability of previous iteration
        """
        self.t += 1
        
        if self.schedule is not None:
            # generate updating weight
            lmbda = self.schedule(self.t)
            
            # update scalling parameter if wanted
            if self.acc_star is not None:
                self.update_scaling(previous_accpept_prob)
    
    def proposal(self, y):
        """
        Returns a sample from the proposal centred at y, and its log-probability
        """
        
        # generate proposal
        proposal = np.random.randn(self.D) * np.sqrt(self.nu2) + y
        diff = proposal - y
        proposal_log_prob = -self.D * np.log(self.nu2) - 0.5 * diff.dot(diff) / self.nu2
        
        # probability of proposing y when would be sitting at proposal is symmetric
        return proposal, proposal_log_prob, proposal_log_prob
    
