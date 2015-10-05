import time

from kameleon_mcmc.kernel.PolynomialKernel import PolynomialKernel

from kameleon_rks.AdaptiveMetropolis import AdaptiveMetropolis
from kameleon_rks.KameleonGaussian import KameleonGaussian
from kameleon_rks.KameleonRKSGaussian import KameleonRKSGaussian
from kameleon_rks.Metropolis import Metropolis
from kameleon_rks.banana import sample_banana, log_banana_pdf, Banana
from kameleon_rks.gaussian_rks import feature_map, sample_basis, \
    gamma_median_heuristic
import matplotlib.pyplot as plt
import numpy as np
from kameleon_rks import mini_mcmc.mini_mcmc
from tools.convergence_stats import min_ess


plt.ion()


np.random.seed(0)

bananicity = 0.1
V = 100

def delayed_banana_log_pdf(x):
    """
    Delayed version of the banana. To simulate target's where the log-pdf consumes
    most of the computational budget.
    """
    time.sleep(0.1)
    return log_banana_pdf(x, bananicity=bananicity, V=V)

# target log pdf function handle
target_log_pdf = delayed_banana_log_pdf
target_log_pdf = lambda x: log_banana_pdf(x, bananicity=bananicity, V=V)
D = 2

# oracle samples: these can e.g. be obtained from a long MCMC run
num_oracle_samples = 1000
Z = sample_banana(N=num_oracle_samples, D=D)

# Kameleon parameters
n = m = 1000  # number of basis vector or history sub-sample size
nu2_kameleon_rks = 30.
nu2_kameleon = 5.3  # tuned to achieve 0.23 for D=2 with 1000 oracle samples
kernel_gamma = gamma_median_heuristic(Z)
# kernel_gamma = 0.5 / (15)**2 # sigma=15
# kernel_gamma = 0.5 / sigma**2
# sigma = np.sqrt(0.5 / kernel_gamma)
print("Using kernel_gamma=%.4f, corresponding sigma=%.2f" % (kernel_gamma, np.sqrt(0.5 / kernel_gamma)))
gamma2 = 0.1
update_kernel_gamma = 1000  # learn kernel parameter
update_kernel_gamma = None  # fixed kernel parameter
update_kernel_gamma_tol = 0.1
update_kernel_gamma_schedule = lambda t: 1. / (t + 1)

if False:
    # visualise kernel matrix approximation for current kernel
    omega, u = sample_basis(D, m, kernel_gamma)
    Phi = feature_map(Z, omega, u)
    plt.imshow(Phi.T.dot(Phi))
    plt.show()

# AM parameters
nu2_am = 0.19  # tuned to achieve 0.23 for D=2 with 1000 oracle samples
# shares gamma2 with Kameleon

# MH parameters
nu2_mh = 10.9

# MCMC parameters
num_iter = 2000
time_budget = 1000
start = np.zeros(D)
start[1] = -10

# adaptation parameters
schedule = None  # fixed scaling
schedule = lambda t: 1. / ((t + 1) ** 0.5)  # learn scaling
acc_star = 0.234

# Kameleon instance, not no schedule here means Kameleon never adapts
# need to perform batch update to update internal covariance
kameleon_rks = KameleonRKSGaussian(D, kernel_gamma, m, nu2_kameleon_rks, gamma2, schedule, acc_star,
                                   update_kernel_gamma, update_kernel_gamma_schedule, update_kernel_gamma_tol)
if schedule is None:
    kameleon_rks.set_batch_covariance(Z)

kameleon = KameleonGaussian(D, kernel_gamma, n, nu2_kameleon, gamma2, schedule, acc_star,
                            update_kernel_gamma, update_kernel_gamma_schedule, update_kernel_gamma_tol)
if schedule is None:
    kameleon.set_oracle_samples(Z)

# AM instance
am = AdaptiveMetropolis(D, nu2_am, gamma2, schedule, acc_star)
if schedule is None:
    am.set_batch_covariance(Z)
    
# MH instance
mh = Metropolis(D, nu2_mh, schedule, acc_star)

# run MCMC
if True:
    # choose which to use
    for sampler in [mh, am, kameleon, kameleon_rks]:
        print(sampler.__class__.__name__)
        np.random.seed(0)
        start_time = time.time()
        results = mini_mcmc(sampler, start, num_iter, target_log_pdf, D, time_budget)
        time_taken = time.time() - start_time
        samples, proposals, log_probs_proposals, log_probs, acceptance_log_probs, accepteds, times = results
    
        ess = min_ess(samples)
        norm_of_mean = np.linalg.norm(np.mean(samples, axis=0))
        mmd = PolynomialKernel(degree=3).estimateMMD(Z, samples[np.random.permutation(len(samples))[:5000]])
        print(sampler.__class__.__name__)
        print("Time taken: %.4f" % time_taken)
        print("min ess: %.2f, norm of mean: %.2f, mmd: %.2e" % (ess, norm_of_mean, mmd))
        print("min ess/time: %.2f, norm of mean * time = %.2f, mmd*time=%.2e" % \
              (ess / time_taken, norm_of_mean * time_taken, mmd * time_taken))
        print("Final nu2: %.2f" % sampler.nu2)
        print("emp quanitles: %s" % str(Banana(bananicity=bananicity, V=V).emp_quantiles(samples)))
        print("\n")
        
        
        plt.figure()
        plt.title(sampler.__class__.__name__)
        plt.plot(samples[:, 0], samples[:, 1])
        plt.xlim([-25, 25])
        plt.ylim([-15, 60])
        
        plt.draw()
