from scipy.spatial.distance import cdist, squareform, pdist

from kameleon_rks.densities.gaussian import sample_gaussian, log_gaussian_pdf
from kameleon_rks.proposals.Metropolis import StaticMetropolis
from kameleon_rks.tools.log import Log
import numpy as np
from old.gaussian_rks import gamma_median_heuristic


logger = Log.get_logger()

def gamma_median_heuristic(Z, num_subsample=1000):
    """
    Computes the median pairwise distance in a random sub-sample of Z.
    Returns a \gamma for k(x,y)=\exp(-\gamma ||x-y||^2), according to the median heuristc,
    i.e. it corresponds to \sigma in k(x,y)=\exp(-0.5*||x-y||^2 / \sigma^2) where
    \sigma is the median distance. \gamma = 0.5/(\sigma^2)
    """
    inds = np.random.permutation(len(Z))[:np.max([num_subsample, len(Z)])]
    dists = squareform(pdist(Z[inds], 'sqeuclidean'))
    median_dist = np.median(dists[dists > 0])
    sigma = np.sqrt(0.5 * median_dist)
    gamma = 0.5 / (sigma ** 2)
    
    return gamma

class OracleKameleon(StaticMetropolis):
    """
    Implements a version of Kameleon MCMC woith oracle samples
    """
    
    def __init__(self, D, target_log_pdf, n, kernel_sigma, step_size, gamma2=0.1, schedule=None, acc_star=0.234):
        
        StaticMetropolis.__init__(self, D, target_log_pdf, step_size, schedule, acc_star)
        
        self.n = n
        self.kernel_sigma = kernel_sigma
        self.gamma2 = gamma2
        
        self.Z = np.zeros((0, D))
        
    def set_batch(self, Z):
        if self.n is not None:
            if len(Z) > self.n:
                inds = np.random.permutation(len(Z))[:self.n]
                self.Z = Z[inds]
        else:
            self.Z = Z
    
    def proposal(self, current, current_log_pdf, **kwargs):
        if self.Z is None and self.schedule is None:
            raise ValueError("%s has not seen data yet. Call set_batch()" % self.__class__.__name__)
        
        if current_log_pdf is None:
            current_log_pdf = self.target_log_pdf(current)
        
        L_R = self._construct_proposal_covariance(current)
        proposal = sample_gaussian(N=1, mu=current, Sigma=L_R, is_cholesky=True)[0]
        proposal_log_prob = log_gaussian_pdf(proposal, current, L_R, is_cholesky=True)
        proposal_log_pdf = self.target_log_pdf(proposal)
        
        # probability of proposing y when would be sitting at proposal
        L_R_inv = self._construct_proposal_covariance(proposal)
        proopsal_log_prob_inv = log_gaussian_pdf(current, proposal, L_R_inv, is_cholesky=True)
        
        log_acc_prob = proposal_log_pdf - current_log_pdf + proopsal_log_prob_inv - proposal_log_prob
        log_acc_prob = np.min([0, log_acc_prob])
        
        results_kwargs = {}
        
        return proposal, proposal_log_pdf, current_log_pdf, proposal_log_prob, proopsal_log_prob_inv, results_kwargs
    
    def _construct_proposal_covariance(self, y):
        """
        Helper method to compute Cholesky factor of the Gaussian Kameleon proposal centred at y.
        """
        R = self.gamma2 * np.eye(self.D)
        
        if len(self.Z) > 0:
            # the code is parametrised in gamma=1./sigma
            kernel_gamma = 1. / self.kernel_sigma
            # k(y,z) = exp(-gamma ||y-z||)
            # d/dy k(y,z) = k(y,z) * (-gamma * d/dy||y-z||^2)
            #             = 2 * k(y,z) * (-gamma * ||y-z||^2)
            #             = 2 * k(y,z) * (gamma * ||z-y||^2)
            
            # gaussian kernel gradient, same as in kameleon-mcmc package, but without python overhead
            sq_dists = cdist(y[np.newaxis, :], self.Z, 'sqeuclidean')
            k = np.exp(-kernel_gamma * sq_dists)
            neg_differences = self.Z - y
            G = 2 * kernel_gamma * (k.T * neg_differences)
            
            # Kameleon
            G *= 2  # = M
            # R = gamma^2 I + \eta^2 * M H M^T
            H = np.eye(len(self.Z)) - 1.0 / len(self.Z)
            R += self.step_size * G.T.dot(H.dot(G))
        
        L_R = np.linalg.cholesky(R)
        
        return L_R
    
class Kameleon(OracleKameleon):
    """
    Implements kernel adaptive StaticMetropolis Hastings.
    """
    
    def __init__(self, D, target_log_pdf, n, kernel_sigma, step_size, gamma2=0.1, schedule=None, acc_star=0.234):
        
        OracleKameleon.__init__(self, D, target_log_pdf, n, kernel_sigma, step_size, gamma2, schedule, acc_star)
    
    def _update_kernel_sigma(self):
        # avoid linalg errors from proposal covariance being rank defficient
        if len(self.Z) >= self.n:
            # re-compute median heuristic for kernel
            self.kernel_sigma = 1. / gamma_median_heuristic(self.Z)

    def update(self, Z, num_new = 1):
        #num_new is not really of interest here
        if self.schedule is not None and len(Z) >= self.n:
            # generate updating probability
            lmbda = self.schedule(self.t)
            
            
            if np.random.rand() < lmbda:
                # update sub-sample of chain history
                OracleKameleon.set_batch(self, Z)
                logger.debug("Updated chain history sub-sample of size %d with probability lmbda=%.3f" % (self.n, lmbda))
                
                self._update_kernel_sigma()
                logger.debug("Re-computed kernel bandwith using median heuristic to sigma=%.3f" % self.kernel_sigma)
