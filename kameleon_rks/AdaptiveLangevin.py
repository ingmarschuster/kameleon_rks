from kameleon_rks.AdaptiveMetropolis import AdaptiveMetropolis
from kameleon_rks.gaussian import sample_gaussian, log_gaussian_pdf


class AdaptiveLangevin(AdaptiveMetropolis):
    """
    Implements adaptive langevin proposal. Performs efficient low-rank updates of Cholesky
    factor of covariance. Covariance itself is not stored/updated, only its Cholesky factor.
    """
    
    def __init__(self, D, grad, nu2, gamma2, eps=1., schedule=None, acc_star=None):
        """
        D             - Input space dimension
        grad          - Function to calculate gradient at any position in support
        nu2           - Scaling parameter for covariance
        gamma2        - Exploration parameter. Added to learned variance
        eps           - Step size parameter for langevin. Like in NUTS/HMC
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
        
        AdaptiveMetropolis.__init__(self, D, nu2, gamma2, schedule, acc_star)
        
        self.grad = grad
        self.eps = eps
    
    def proposal(self, y):
        """
        Returns a sample from the proposal centred at y, and its log-probability
        """
        # generate proposal
        forw_mu = y + self.L_C.dot(self.L_C.T).dot(self.eps*self.grad(y))
        proposal = sample_gaussian(N=1, mu=forw_mu, Sigma=self.L_C, is_cholesky=True)[0]
        forw_log_prob = log_gaussian_pdf(proposal, forw_mu, self.L_C, is_cholesky=True)
        
        backw_mu = proposal + self.L_C.dot(self.L_C.T).dot(self.eps*self.grad(proposal))
        backw_log_prob = log_gaussian_pdf(proposal, backw_mu, self.L_C, is_cholesky=True)
        
        
        # probability of proposing y when would be sitting at proposal is symmetric
        return proposal, forw_log_prob, backw_log_prob
    
