from kameleon_rks.densities.banana import log_banana_pdf, sample_banana
from kameleon_rks.examples.plotting import visualise_trace
from kameleon_rks.mcmc.mini_mcmc import mini_mcmc
from kameleon_rks.proposals.Kameleon import StaticKameleon, AdaptiveKameleon,\
    gamma_median_heuristic
from kameleon_rks.proposals.Langevin import StaticLangevin, AdaptiveLangevin,\
    StaticKernelLangevin, AdaptiveKernelLangevin
from kameleon_rks.proposals.Metropolis import StaticMetropolis, \
    AdaptiveMetropolis
from kameleon_rks.tools.log import Log
from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian,\
    KernelExpLiteGaussianAdaptive
import matplotlib.pyplot as plt
import numpy as np


def one_over_sqrt_t_schedule(t):
    return 1. / np.sqrt(1 + t)

def get_StaticMetropolis_instance(D, target_log_pdf):
    
    step_size = 8.
    schedule = one_over_sqrt_t_schedule
    acc_star = 0.234
    instance = StaticMetropolis(D, target_log_pdf, step_size, schedule, acc_star)
    
    return instance

def get_AdaptiveMetropolis_instance(D, target_log_pdf):
    
    step_size = 8.
    schedule = one_over_sqrt_t_schedule
    acc_star = 0.234
    gamma2 = 0.1
    instance = AdaptiveMetropolis(D, target_log_pdf, step_size, gamma2, schedule, acc_star)
    
    return instance

def get_StaticKameleon_instance(D, target_log_pdf):
    
    step_size = 30.
    schedule = one_over_sqrt_t_schedule
    acc_star = 0.234
    gamma2 = 0.1
    n = 500
    
    Z = sample_banana(N=n, D=D, bananicity=0.03, V=100)
    
    kernel_sigma = 1. / gamma_median_heuristic(Z)
    instance = StaticKameleon(D, target_log_pdf, n, kernel_sigma, step_size, gamma2, schedule, acc_star)
    instance.set_batch(Z)
    
    return instance

def get_AdaptiveKameleon_instance(D, target_log_pdf):
    
    step_size = 1.
    schedule = one_over_sqrt_t_schedule
    acc_star = 0.234
    gamma2 = 0.1
    
    kernel_sigma = 1.
    n = 500

    instance = AdaptiveKameleon(D, target_log_pdf, n, kernel_sigma, step_size, gamma2, schedule, acc_star)
    
    return instance

def get_StaticLangevin_instance(D, target_log_pdf, target_grad):
    
    step_size = 1.
    schedule = one_over_sqrt_t_schedule
    acc_star = 0.234
    
    instance = StaticLangevin(D, target_log_pdf, target_grad, step_size, schedule, acc_star)
    
    return instance

def get_AdaptiveLangevin_instance(D, target_log_pdf, target_grad):
    
    step_size = 1.
    schedule = one_over_sqrt_t_schedule
    acc_star = 0.234
    gamma2 = 0.1
    
    instance = AdaptiveLangevin(D, target_log_pdf, target_grad, step_size, gamma2, schedule, acc_star)
    
    return instance

def get_StaticKernelLangevin_instance(D, target_log_pdf, target_grad):
    
    step_size = 1.
    schedule = one_over_sqrt_t_schedule
    acc_star = 0.234
    N=500
    Z = sample_banana(N=N, D=D, bananicity=0.03, V=100)
    
    surrogate = KernelExpLiteGaussianAdaptive(sigma=1, lmbda=0.001, D=D, N=N)
    surrogate.fit(Z)
    
    instance = StaticKernelLangevin(D, target_log_pdf, surrogate, step_size, schedule, acc_star)
    
    return instance

def get_AdaptiveKernelLangevin_instance(D, target_log_pdf, target_grad):
    
    step_size = 1.
    schedule = one_over_sqrt_t_schedule
    acc_star = 0.234
    gamma2 = 0.1
    n=500
    
    surrogate = KernelExpLiteGaussian(sigma=1, lmbda=0.001, D=D, N=n)
    
    instance = AdaptiveKernelLangevin(D, target_log_pdf, n, surrogate, step_size, gamma2, schedule, acc_star)
    
    return instance

if __name__ == '__main__':
    Log.set_loglevel(10)
    D = 2
    
    bananicity = 0.03
    V = 100
    target_log_pdf = lambda x: log_banana_pdf(x, bananicity, V, compute_grad=False)
    target_grad = lambda x: log_banana_pdf(x, bananicity, V, compute_grad=True)

    samplers = [
                get_StaticMetropolis_instance(D, target_log_pdf),
                get_AdaptiveMetropolis_instance(D, target_log_pdf),
                get_StaticKameleon_instance(D, target_log_pdf),
                get_AdaptiveKameleon_instance(D, target_log_pdf),
                get_StaticLangevin_instance(D, target_log_pdf, target_grad),
                get_AdaptiveLangevin_instance(D, target_log_pdf, target_grad),
                get_StaticKernelLangevin_instance(D, target_log_pdf, target_grad),
                get_StaticKernelLangevin_instance(D, target_log_pdf, target_grad),
                get_AdaptiveKernelLangevin_instance(D, target_log_pdf, target_grad),
                
                ]

    for sampler in samplers:
        # MCMC parameters, feel free to increase number of iterations
        start = np.zeros(D)
        num_iter = 1000
        
        # run MCMC
        samples, proposals, accepted, acc_prob, log_pdf, times, step_sizes = mini_mcmc(sampler, start, num_iter, D)
        
        visualise_trace(samples, log_pdf, accepted, step_sizes)
        plt.suptitle("%s, acceptance rate: %.2f" % \
                     (sampler.__class__.__name__, np.mean(accepted)))
        
    plt.show()
