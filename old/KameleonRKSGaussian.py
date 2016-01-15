import collections

from kameleon_rks.densities.gaussian import sample_gaussian, log_gaussian_pdf
from old.gaussian_rks import sample_basis, feature_map, \
    feature_map_grad_single, feature_map_single, gamma_median_heuristic
import numpy as np


class KameleonRKSGaussian():
    """
    Implements a random kitchen sink version of Kameleon MCMC.
    """
    
    def __init__(self, D, kernel_gamma, m, step_size, gamma2=0.1, schedule=None, acc_star=0.234,
                 update_kernel_gamma=None, update_kernel_gamma_schedule=None, update_kernel_gamma_tol=0.1):
        """
        D                            - Input space dimension
        kernel_gamma                 - Gaussian kernel parameter
        m                            - Feature space dimension
        gamma2                       - Exploration parameter. Kameleon falls back
                                       to random walk with that scaling when in unexplored regions.
                                       Increasing increases exploration but decreases mixing in explored regions.
        step_size                         - Gradient step size. Effectively a scaling parameter.
        schedule                     - Optional. Function that generates adaptation weights
                                       given the MCMC iteration number.
                                       The weights are used in the stochastic updating of the
                                       feature covariance.
                                       If not set, feature covariance is never updated. In that case, call
                                       batch_covariance() before using. 
        acc_star                       Optional: If set, the step_size parameter is tuned so that
                                       average acceptance equals eta_star, using the same schedule
                                       as for the covariance update (If schedule is set, otherwise
                                       ignored)
        update_kernel_gamma          - Optional. If set to an integer, collects a sliding
                                       window of past samples in the update() method.
                                       Uses 1./t as probability to re-set the
                                       kernel bandwidth via computing the media distance
                                       in the collected samples. Then updates the random
                                       feature basis.
                                       The window size depends on the support of the distribution,
                                       and in particular the number of samples to estimate the median
                                       distance reliably. Set to a few thousand if in doubt.
                                       Suggested to only use in pilot runs and then fix the found
                                       kernel_gamma.
                                       Note that a separate updating schedule can be provided.
        update_kernel_gamma_schedule - Optional. A schedule different to 1./t can be provided to
                                       update the kernel bandwidth.
        update_kernel_gamma_tol      - Optional. Tolerance for kernel parameter update.
                                       Bandwidth is only updated if at least that far from previous
        """
        self.kernel_gamma = kernel_gamma
        self.m = m
        self.D = D
        self.gamma2 = gamma2
        self.step_size = step_size
        self.schedule = schedule
        self.acc_star = acc_star
        self.update_kernel_gamma = update_kernel_gamma
        self.update_kernel_gamma_schedule = update_kernel_gamma_schedule
        self.update_kernel_gamma_tol = update_kernel_gamma_tol
        
        # scaling parameter evolution might be collected to assess convergence
        self.nu2s = [step_size]
        
        # some sanity checks
        if acc_star is not None:
            assert acc_star > 0 and acc_star < 1
            
        if schedule is not None:
            lmbdas = np.array([schedule(t) for t in  np.arange(100)])
            assert np.all(lmbdas > 0)
            assert np.allclose(np.sort(lmbdas)[::-1], lmbdas)
            
        if self.update_kernel_gamma:
            self.past_samples = collections.deque()
            
        if self.update_kernel_gamma_schedule is not None:
            lmbdas = np.array([update_kernel_gamma_schedule(t) for t in  np.arange(100)])
            assert np.all(lmbdas > 0)
            assert np.allclose(np.sort(lmbdas)[::-1], lmbdas)
        
        self._initialise()
    
    def _initialise(self):
        """
        Initialises internal state. To be called before MCMC chain starts.
        """
        # fix feature space random basis
        self.omega, self.u = sample_basis(self.D, self.m, self.kernel_gamma)
        
        # _initialise running averages for feature covariance
        self.t = 0

        if self.schedule is not None:
            # start from scratch
            self.mu = np.zeros(self.m)
            
            # _initialise as isotropic
            self.C = np.eye(self.m)
        else:
            # make user call the set_batch_covariance() function
            self.mu = None
            self.C = None

    def set_batch_covariance(self, Z):
        Phi = feature_map(Z, self.omega, self.u)
        self.mu = np.mean(Phi, axis=0)
        self.C = np.cov(Phi.T)
    
    def update_step_size(self, accept_prob):
        # generate learning rate
        lmbda = self.schedule(self.t)
        
        # difference desired and actuall acceptance rate
        diff = accept_prob - self.acc_star
            
        self.step_size = np.exp(np.log(self.step_size) + lmbda * diff)

    def next_iteration(self):
        self.t += 1
        

    def update(self, z_new, previous_accpept_prob):
        """
        Updates the proposal covariance and potentially scaling parameter, according to schedule.
        Note that every call increases a counter that is used for the schedule (if set)
        
        If not schedule is set, this method does not have any effect unless counting.
        
        Parameters:
        z_new                       - A 1-dimensional array of size (D) of.
        previous_accpept_prob       - Acceptance probability of previous iteration
        """
        self.next_iteration()
        
        if self.schedule is not None:
            # generate updating weight
            lmbda = self.schedule(self.t)
            
            # project current point
            phi = feature_map_single(z_new, self.omega, self.u)
            
            # update
            centred = self.mu - phi
            self.mu = self.mu * (1 - lmbda) + lmbda * phi
            self.C = self.C * (1 - lmbda) + lmbda * np.outer(centred, centred)
            
            # update scalling parameter if wanted
            if self.acc_star is not None:
                self.update_step_size(previous_accpept_prob)
                self.nu2s.append(self.step_size)
            
            if self.update_kernel_gamma is not None:
                # update sliding window
                self.past_samples.append(z_new)
                if len(self.past_samples) > self.update_kernel_gamma:
                    self.past_samples.popleft()
                
                num_samples_window = len(self.past_samples)
                
                # probability of updating
                if self.update_kernel_gamma_schedule is not None:
                    update_prob = self.update_kernel_gamma_schedule(self.t)
                else:
                    update_prob = 1. / (self.t + 1)
                
                # update kernel bandwidth (if window full yet)
                if np.random.rand() < update_prob and num_samples_window >= self.update_kernel_gamma:
                    
                    # transform past samples into array
                    Z = np.array(self.past_samples)
                    
                    # compute new kernel gamma
                    print("Checking whether to update kernel_gamma")
                    new_kernel_gamma = gamma_median_heuristic(Z, num_samples_window)
                    diff = np.abs(new_kernel_gamma - self.kernel_gamma)
                    
                    # only update if change above tolerance
                    if np.abs(diff > self.update_kernel_gamma_tol):
                        self.kernel_gamma = new_kernel_gamma
                        
                        # re-sample basis
                        self.omega, self.u = sample_basis(self.D, self.m, self.kernel_gamma)
                        
                        # populate feature covariance from past samples
                        self.set_batch_covariance(Z)
                        
                        print("Updated kernel gamma to %.3f (from %d samples)" % (self.kernel_gamma, num_samples_window))
    
    def proposal(self, current):
        """
        Returns a sample from the proposal centred at current, and its log-probability
        """
        
        if self.schedule is None and (self.mu is None or self.C is None):
            raise ValueError("Kameleon has not seen data yet." \
                             "Either call set_batch_covariance() or set update schedule")
        
        L_R = self._construct_proposal_covariance(current)
        proposal = sample_gaussian(N=1, mu=current, Sigma=L_R, is_cholesky=True)[0]
        proposal_log_prob = log_gaussian_pdf(proposal, current, L_R, is_cholesky=True)
        
        # probability of proposing current when would be sitting at proposal
        L_R_inv = self._construct_proposal_covariance(proposal)
        proopsal_log_prob_inv = log_gaussian_pdf(current, proposal, L_R_inv, is_cholesky=True)
        
        return proposal, proposal_log_prob, proopsal_log_prob_inv
    
    def _construct_proposal_covariance(self, y):
        """
        Helper method to compute Cholesky factor of the Gaussian Kameleon-lite proposal centred at y.
        """
        # compute gradient projection
        grad_phi_y = feature_map_grad_single(y, self.omega, self.u)
        
        # construct covariance, adding exploration noise
        R = self.gamma2 * np.eye(self.D) + self.step_size * np.dot(grad_phi_y, (self.m ** 2) * np.dot(self.C, grad_phi_y.T))
        L_R = np.linalg.cholesky(R)
        
        return L_R
    
