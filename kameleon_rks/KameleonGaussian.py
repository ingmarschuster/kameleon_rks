from scipy.spatial.distance import cdist

from kameleon_rks.gaussian import sample_gaussian, log_gaussian_pdf
from kameleon_rks.gaussian_rks import gamma_median_heuristic
import numpy as np


class KameleonGaussian():
    """
    Implements a random kitchen sink version of Kameleon MCMC, using a random sub-sample of the chain history.
    """
    
    def __init__(self, D, kernel_gamma, n, nu2, gamma2=0.1, schedule=None, acc_star=0.234,
                 update_kernel_gamma=False, update_kernel_gamma_schedule=None, update_kernel_gamma_tol=0.1):
        """
        D                            - Input space dimension
        kernel_gamma                 - Gaussian kernel parameter
        n                            - Number of sub-samples of the chain history
        gamma2                       - Exploration parameter. Kameleon falls back
                                       to random walk with that scaling when in unexplored regions.
                                       Increasing increases exploration but decreases mixing in explored regions.
        nu2                         - Gradient step size. Effectively a scaling parameter.
        schedule                     - Optional. Function that generates adaptation probabilities
                                       given the MCMC iteration number.
                                       The probabilities are used in the updating updating of the
                                       random sub-sample of the chain history.
                                       If not set, chain history is never updated. In that case, call
                                       set_oracle_samples() before using. 
        acc_star                       Optional: If set, the nu2 parameter is tuned so that
                                       average acceptance equals acc_star, using the same schedule
                                       as for the chain history update (If schedule is set, otherwise
                                       ignored)
        update_kernel_gamma          - Optional. If set to True, updates kernel bandwidth.
                                       Uses 1./t as probability to re-set the
                                       kernel bandwidth via computing the media distance
                                       in the collected (subsampled) samples.
                                       Suggested to only use in pilot runs and then fix the found
                                       kernel_gamma.
                                       Note that a separate updating schedule can be provided.
        update_kernel_gamma_schedule - Optional. A schedule different to 1./t can be provided to
                                       update the kernel bandwidth.
        update_kernel_gamma_tol      - Optional. Tolerance for kernel parameter update.
                                       Bandwidth is only updated if at least that far from previous
        """
        self.kernel_gamma = kernel_gamma
        self.n = n
        self.D = D
        self.gamma2 = gamma2
        self.nu2 = nu2
        self.schedule = schedule
        self.acc_star = acc_star
        self.update_kernel_gamma = update_kernel_gamma
        self.update_kernel_gamma_schedule = update_kernel_gamma_schedule
        self.update_kernel_gamma_tol = update_kernel_gamma_tol
        
        # scaling parameter evolution might be collected to assess convergence
        self.nu2s = [nu2]
        
        # some sanity checks
        if acc_star is not None:
            assert acc_star > 0 and acc_star < 1
            
        if schedule is not None:
            lmbdas = np.array([schedule(t) for t in  np.arange(100)])
            assert np.all(lmbdas > 0)
            assert np.allclose(np.sort(lmbdas)[::-1], lmbdas)
            
        if self.update_kernel_gamma_schedule is not None:
            lmbdas = np.array([update_kernel_gamma_schedule(t) for t in  np.arange(100)])
            assert np.all(lmbdas > 0)
            assert np.allclose(np.sort(lmbdas)[::-1], lmbdas)
        
        self.initialise()
    
    def initialise(self):
        """
        Initialises internal state. To be called before MCMC chain starts.
        """
        self.t = 0
        
        # initialise chain history and random sub-sample
        self.chain_history = []
        if self.schedule is not None:
            self.Z = np.zeros((0, self.D))
        else:
            # make user call the set_oracle_samples() function
            self.Z = None

    def set_oracle_samples(self, Z):
        self.Z = Z
    
    def set_batch_covariance(self, Z):
        self.set_oracle_samples(Z)
    
    def update_scaling(self, accept_prob):
        assert(self.schedule is not None)
        self.nu2 = np.exp(np.log(self.nu2) + self.schedule(self.t) * (accept_prob - self.acc_star))

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
        self.t += 1
        self.chain_history.append(z_new)
        
        if self.schedule is not None:
            # generate updating probability
            lmbda = self.schedule(self.t)
            
            # update chain history
            if np.random.rand() < lmbda:
                num_samples_Z = np.min([self.n, self.t])
                inds = np.random.permutation(self.t)[:num_samples_Z]
                self.Z = np.zeros((num_samples_Z, self.D))
                for i, ind in enumerate(inds):
                    self.Z[i] = self.chain_history[ind]
            
            # update scaling parameter if wanted
            if self.acc_star is not None:
                diff = previous_accpept_prob - self.acc_star
                self.nu2 = np.exp(np.log(self.nu2) + lmbda * diff)
                self.nu2s.append(self.nu2)
            
            # update kernel parameter if history contains at least n samples
            if self.update_kernel_gamma and self.t >= self.n:
                # probability of updating
                if self.update_kernel_gamma_schedule is not None:
                    update_prob = self.update_kernel_gamma_schedule(self.t)
                else:
                    update_prob = 1. / (self.t + 1)
                
                # update kernel bandwidth (if window full yet)
                if np.random.rand() < update_prob:
                    # compute new kernel gamma
                    print("Checking whether to update kernel_gamma")
                    new_kernel_gamma = gamma_median_heuristic(self.Z, self.n)
                    diff = np.abs(new_kernel_gamma - self.kernel_gamma)
                    
                    # only update if change above tolerance
                    if np.abs(diff > self.update_kernel_gamma_tol):
                        self.kernel_gamma = new_kernel_gamma
                        
                        print("Updated kernel gamma to %.3f" % self.kernel_gamma)
    
    def proposal(self, y):
        """
        Returns a sample from the proposal centred at y, and its log-probability
        """
        
        if self.schedule is None and self.Z is None:
            raise ValueError("Kameleon has not seen data yet." \
                             "Either call set_oracle_samples() or set update schedule")
        
        L_R = self.construct_proposal_covariance_(y)
        proposal = sample_gaussian(N=1, mu=y, Sigma=L_R, is_cholesky=True)[0]
        proposal_log_prob = log_gaussian_pdf(proposal, y, L_R, is_cholesky=True)
        
        # probability of proposing y when would be sitting at proposal
        L_R_inv = self.construct_proposal_covariance_(proposal)
        proopsal_log_prob_inv = log_gaussian_pdf(y, proposal, L_R_inv, is_cholesky=True)
        
        return proposal, proposal_log_prob, proopsal_log_prob_inv
    
    def construct_proposal_covariance_(self, y):
        """
        Helper method to compute Cholesky factor of the Gaussian Kameleon proposal centred at y.
        """
        R = self.gamma2 * np.eye(self.D)
        
        if len(self.Z) > 0:
            # k(y,z) = exp(-gamma ||y-z||)
            # d/dy k(y,z) = k(y,z) * (-gamma * d/dy||y-z||^2)
            #             = 2 * k(y,z) * (-gamma * ||y-z||^2)
            #             = 2 * k(y,z) * (gamma * ||z-y||^2)
            
            # gaussian kernel gradient, same as in kameleon-mcmc package, but without python overhead
            sq_dists = cdist(y[np.newaxis, :], self.Z, 'sqeuclidean')
            k = np.exp(-self.kernel_gamma * sq_dists)
            neg_differences = self.Z - y
            G = 2 * self.kernel_gamma * (k.T * neg_differences)
            
            # Kameleon
            G *= 2  # = M
            # R = gamma^2 I + \eta^2 * M H M^T
            H = np.eye(len(self.Z)) - 1.0 / len(self.Z)
            R += self.nu2 * G.T.dot(H.dot(G))
            
        L_R = np.linalg.cholesky(R)
        
        return L_R
    
