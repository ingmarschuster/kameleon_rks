from kameleon_rks.gaussian import sample_gaussian, log_gaussian_pdf
from kameleon_rks.gaussian_rks import sample_basis, feature_map,\
    feature_map_grad_single
from kameleon_rks.running_averages import rank_m_update_mean_covariance
import numpy as np

class KameleonRKSGaussian():
    """
    Implements a random kitchen sink version of Kameleon MCMC.
    """
    
    def __init__(self, kernel_gamma, m, D, gamma2, eta2, schedule=None):
        """
        kernel_gamma  - Gaussian kernel parameter
        m             - Feature space dimension
        D             - Input space dimension
        gamma2        - Exploration parameter
        eta2          - Gradient step size
        schedule      - Optional. Function that generates adaptation probablities as a
                        function of MCMC iteration. Affects update() function
        """
        self.kernel_gamma = kernel_gamma
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
        self.omega, self.u = sample_basis(self.D, self.m, self.kernel_gamma)
        
        # initialise running averages for feature covariance: t, mean, and n*covariance
        self.n = None
        self.mu = None
        self.M2 = None
        self.C = None
        
        # transition kernel tranistion t
        self.t = 0
        
        # iteration of last proposal update
        self.last_update = 0

    def update(self, z_new):
        """
        Updates the Kameleon-lite proposal. If a schedule was set, it is used as update probability.
        Note that every call increases a counter that is used for the schedule (if set)
        
        If a schedule is set, updates the proposal mechanism according to schedule probability.
        
        z_new is a 1-dimensional array of size (D) of.
        """
        self.t += 1
        
        if self.schedule is not None:
            # update according to schedule and all points since last update
            if np.random.rand() < self.schedule(self.t):
                self.update_feature_covariance_(z_new)
        else:
            # always update using last point
            self.update_feature_covariance_(z_new)
        
    
    def proposal(self, y):
        """
        Returns a sample from the proposal centred at y, and its log-probability
        """
        L_R = self.construct_proposal_covariance_(y)
        proposal = sample_gaussian(N=1, mu=y, Sigma=L_R, is_cholesky=True)[0]
        proposal_log_prob = log_gaussian_pdf(proposal, y, L_R, is_cholesky=True)
        
        return proposal, proposal_log_prob
    
    def update_feature_covariance_(self, Z_new):
        """
        Helper method to update the feature space mean and covariance.
        
        Z_new is assumed to be either a 2d array or a 1d array
        """
        
        # ignore empty updates
        if len(Z_new)==0:
            return
        
        # make sure array dimension is correct
        if Z_new.ndim==1:
            Z_new = Z_new[np.newaxis,:]
        
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
    