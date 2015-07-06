from kameleon_rks.gaussian import sample_gaussian, log_gaussian_pdf
from kameleon_rks.gaussian_rks import sample_basis, feature_map,\
    feature_map_grad_single
from kameleon_rks.running_averages import rank_m_update_mean_covariance
import numpy as np

class KameleonRKSGaussian():
    """
    Implements a random kitchen sink version of Kameleon MCMC.
    """
    
    def __init__(self, kernel_width, m, D, gamma2, eta2, schedule=None):
        """
        kernel_width  - Gaussian kernel parameter
        m             - Feature space dimension
        D             - Input space dimension
        gamma2        - Exploration parameter
        eta2          - Gradient step size
        schedule      - Function that generates adaptation probablities as a
                        function of MCMC iteration.
        """
        self.kernel_width = kernel_width
        self.m = m
        self.D = D
        self.gamma2 = gamma2
        self.eta2 = eta2
        self.schedule = schedule
    
    def initialise(self):
        """
        Initialises internal state. To be called before MCMC chain starts.
        """
        # fix feature space random basis
        self.omega, self.u = sample_basis(self.D, self.m, self.kernel_width)
        
        # initialise running averages for feature covariance: t, mean, and n*covariance
        self.n = None
        self.mu = None
        self.M2 = None
        self.C = None
        
        # transition kernel tranistion t
        self.t = 0
        
        # iteration of last proposal update
        self.last_update = 0
        
        # learning rate as anonymous function if not set
        if self.schedule is None:
            self.schedule = lambda t: 1./t**.5

    def update(self, MCMC_trajectory):
        """
        Updates the Kameleon-lite proposal with a certain probability.
        To be called *every* MCMC iteration.
        
        MCMC_trajectory are all samples from the MCMC chain as a 2-dimensional array of size (txD)
        """
        self.t += 1
        
        # decide whether to update proposal
        if np.random.rand() < self.schedule(self.t):
            Z_new = MCMC_trajectory[self.last_update:self.t]
            self.last_update = self.t
            
            self.update_feature_covariance_(Z_new)
    
    def proposal(self, y):
        """
        Returns a sample from the proposal centred at y.
        """
        L_R = self.construct_proposal_covariance_(y)
        proposal = sample_gaussian(N=1, mu=y, Sigma=L_R, is_cholesky=True)[0]
        proposal_log_prob = log_gaussian_pdf(proposal, y, L_R, is_cholesky=True)
        
        return proposal, proposal_log_prob
    
    def update_feature_covariance_(self, Z_new):
        """
        Helper method to update the feature space mean and covariance.
        """
        Phi_new = feature_map(Z_new, self.omega, self.u)
        self.mu, self.C, self.n, self.M2 = rank_m_update_mean_covariance(Phi_new, self.n, self.mu, self.M2)
    
    def construct_proposal_covariance_(self, y):
        """
        Helper method to compute Cholesky factor of the Gaussian Kameleon-lite proposal centred at y.
        """
        # compute gradient projection
        grad_phi_y = feature_map_grad_single(y, self.omega, self.u)
        
        # construct covariance, adding exploration noise
        R = self.gamma2 * np.eye(self.D) + self.eta2 * np.dot(grad_phi_y, np.dot(self.C, grad_phi_y.T))
        L_R = np.linalg.cholesky(R)
        
        return L_R
    