from kameleon_rks.densities.banana import log_banana_pdf, sample_banana
from kameleon_rks.densities.gaussian import mvnorm
from kameleon_rks.examples.plotting import visualize_scatter
from kameleon_rks.proposals.Langevin import AdaptiveLangevin, \
    OracleKernelAdaptiveLangevin, KernelAdaptiveLangevin
from kameleon_rks.proposals.Metropolis import AdaptiveMetropolis
from kameleon_rks.samplers.mini_smc import mini_smc
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
    instance.fixed_step_size = True
    
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
    
    return instance

if __name__ == '__main__':
    Log.set_loglevel(10)
    D = 2
    
    bananicity = 0.03
    V = 100
    
    target_log_pdf = lambda x: log_banana_pdf(x, bananicity, V, compute_grad=False)
    grad = lambda x: log_banana_pdf(x, bananicity, V, compute_grad=True)

    samplers = [
                get_AdaptiveMetropolis_instance(D, target_log_pdf),
                get_AdaptiveLangevin_instance(D, target_log_pdf, grad),
                get_OracleKernelAdaptiveLangevin_instance(D, target_log_pdf, grad),
                get_KernelAdaptiveLangevin_instance(D, target_log_pdf, grad),
                ]
    
    for sampler in samplers:
        bridge_start = mvnorm(np.zeros(D), np.eye(D) * np.sqrt(2.8 / D))
        
        num_population = 1000
        num_samples = num_population
        samples, log_target_densities, step_sizes, evid = mini_smc(num_samples,
                                                              num_population,
                                                              bridge_start,
                                                              target_log_pdf,
                                                              sampler,
                                                              targ_ef_bridge=0.8)
        visualize_scatter(samples, step_sizes)
        
        plt.suptitle("%s, acceptance rate" % \
                     (sampler.__class__.__name__))
        
    plt.show()

