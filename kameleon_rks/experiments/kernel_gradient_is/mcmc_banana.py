from kameleon_rks.densities.banana import log_banana_pdf, sample_banana
from kameleon_rks.examples.plotting import visualise_trace
from kameleon_rks.proposals.Langevin import AdaptiveLangevin, \
    OracleKernelAdaptiveLangevin, KernelAdaptiveLangevin
from kameleon_rks.proposals.Metropolis import AdaptiveMetropolis
from kameleon_rks.samplers.mini_mcmc import mini_mcmc
from kameleon_rks.tools.convergence_stats import mmd_to_benchmark_sample,\
    min_ess
from kameleon_rks.tools.log import Log
from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
import matplotlib.pyplot as plt
import numpy as np


def one_over_sqrt_t_schedule(t):
    return 1. / np.sqrt(1 + t)

def get_AdaptiveMetropolis_instance(D, target_log_pdf):
    
    step_size = 8.
    schedule = one_over_sqrt_t_schedule
    acc_star = 0.234
    gamma2 = 0.1
    instance = AdaptiveMetropolis(D, target_log_pdf, step_size, gamma2, schedule, acc_star)
    
    return instance

def get_AdaptiveLangevin_instance(D, target_log_pdf, grad):
    
    step_size = 1.
    schedule = one_over_sqrt_t_schedule
    acc_star = 0.574
    gamma2 = 0.1
    
    instance = AdaptiveLangevin(D, target_log_pdf, grad, step_size, gamma2, schedule, acc_star)
    
#     instance.manual_gradient_step_size = 1.
    
    return instance

def get_OracleKernelAdaptiveLangevin_instance(D, target_log_pdf, grad):
    
    step_size = 1.
    schedule = one_over_sqrt_t_schedule
    acc_star = 0.574
    N = 500
    gamma2 = 0.1
    Z = sample_banana(N=N, D=D, bananicity=0.03, V=100)
    
    surrogate = KernelExpLiteGaussian(sigma=10, lmbda=0.001, D=D, N=N)
    surrogate.fit(Z)
    
    instance = OracleKernelAdaptiveLangevin(D, target_log_pdf, N, surrogate, step_size, gamma2, schedule, acc_star)
    
    instance.fixed_step_size = True
    instance.manual_gradient_step_size = 1.
    
    return instance

def get_KernelAdaptiveLangevin_instance(D, target_log_pdf, grad):
    step_size = 1.
    schedule = lambda t : 1. / (t + 1) ** 0.75
    acc_star = 0.574
    gamma2 = 0.1
    n = 500
    
    surrogate = KernelExpLiteGaussian(sigma=10, lmbda=0.001, D=D, N=n)
    instance = KernelAdaptiveLangevin(D, target_log_pdf, n, surrogate, step_size, gamma2, schedule, acc_star)
    
    instance.fixed_step_size = True
    instance.manual_gradient_step_size = 1.
    
    return instance

if __name__ == '__main__':
    Log.set_loglevel(20)
    D = 2
    
    bananicity = 0.03
    V = 100
    benchmark_sample = sample_banana(5000, D, bananicity, V)
    
    target_log_pdf = lambda x: log_banana_pdf(x, bananicity, V, compute_grad=False)
    grad = lambda x: log_banana_pdf(x, bananicity, V, compute_grad=True)

    samplers = [
                get_AdaptiveMetropolis_instance(D, target_log_pdf),
                get_AdaptiveLangevin_instance(D, target_log_pdf, grad),
#                 get_OracleKernelAdaptiveLangevin_instance(D, target_log_pdf, grad),
#                 get_KernelAdaptiveLangevin_instance(D, target_log_pdf, grad),
                ]
    for sampler in samplers:
        start = np.zeros(D)
        num_iter = 10000
        
        # run MCMC
        samples, proposals, accepted, acc_prob, log_pdf, times, step_sizes = mini_mcmc(sampler, start, num_iter, D)
        visualise_trace(samples, log_pdf, accepted, step_sizes)
        plt.suptitle("%s, acceptance rate: %.2f" % \
                     (sampler.__class__.__name__, np.mean(accepted)))
        
        thinning_factor = 1
        thinning_inds = np.arange(len(samples)/2)
        thinning_inds += len(thinning_inds)
        thinning_inds = thinning_inds[np.arange(0,len(thinning_inds),step=thinning_factor)]
        thinned = samples[thinning_inds]
        
        print "MMD", mmd_to_benchmark_sample(thinned, benchmark_sample, degree=3)
        print "ESS", min_ess(thinned) 
        
    if False or True:
        plt.show()

