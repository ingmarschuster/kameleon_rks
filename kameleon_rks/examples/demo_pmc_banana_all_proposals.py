from kameleon_rks.densities.banana import sample_banana, log_banana_pdf
from kameleon_rks.examples.plotting import visualize_scatter
from kameleon_rks.proposals.Kameleon import gamma_median_heuristic,\
    OracleKameleon, Kameleon
from kameleon_rks.proposals.Langevin import StaticLangevin, AdaptiveLangevin,\
    OracleKernelAdaptiveLangevin, KernelAdaptiveLangevin
from kameleon_rks.proposals.Metropolis import StaticMetropolis,\
    AdaptiveMetropolis
from kameleon_rks.samplers.mini_pmc import mini_pmc
from kameleon_rks.tools.log import Log
from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
import matplotlib.pyplot as plt
import numpy as np


def one_over_4th_root_t_schedule(t):
    return 1. / (1 + t) ** (0.24)

def get_StaticMetropolis_instance(D, target_log_pdf):
    
    step_size = 8.
    schedule = one_over_4th_root_t_schedule
    acc_star = 0.234
    instance = StaticMetropolis(D, target_log_pdf, step_size, schedule, acc_star)
    
    return instance

def get_AdaptiveMetropolis_instance(D, target_log_pdf):
    
    step_size = 1.
    schedule = one_over_4th_root_t_schedule
    acc_star = 0.234
    gamma2 = 0.1
    instance = AdaptiveMetropolis(D, target_log_pdf, step_size, gamma2, schedule, acc_star)
    
    return instance

def get_OracleKameleon_instance(D, target_log_pdf):
    
    step_size = 30.
    schedule = one_over_4th_root_t_schedule
    acc_star = 0.234
    gamma2 = 0.1
    n = 500
    
    Z = sample_banana(N=n, D=D, bananicity=0.03, V=100)
    
    kernel_sigma = 1. / gamma_median_heuristic(Z)
    instance = OracleKameleon(D, target_log_pdf, n, kernel_sigma, step_size, gamma2, schedule, acc_star)
    instance.set_batch(Z)
    
    return instance
    
def get_Kameleon_instance(D, target_log_pdf):
    
    step_size = 1.
    schedule = one_over_4th_root_t_schedule
    acc_star = 0.234
    gamma2 = 0.1
    
    kernel_sigma = 1.
    n = 500

    instance = Kameleon(D, target_log_pdf, n, kernel_sigma, step_size, gamma2, schedule, acc_star)
    
    return instance

def get_StaticLangevin_instance(D, target_log_pdf, target_grad):
    
    step_size = 1.
    schedule = one_over_4th_root_t_schedule
    acc_star = 0.234
    
    instance = StaticLangevin(D, target_log_pdf, target_grad, step_size, schedule, acc_star)
    
    return instance

def get_AdaptiveLangevin_instance(D, target_log_pdf, target_grad):
    
    step_size = 1.
    schedule = one_over_4th_root_t_schedule
    acc_star = 0.234
    gamma2 = 0.1
    
    instance = AdaptiveLangevin(D, target_log_pdf, target_grad, step_size, gamma2, schedule, acc_star)
    
    return instance

def get_OracleKernelAdaptiveLangevin_instance(D, target_log_pdf, grad):
    
    step_size = 1.
    schedule = one_over_4th_root_t_schedule
    acc_star = 0.574
    N = 500
    gamma2 = 0.1
    Z = sample_banana(N=N, D=D, bananicity=0.03, V=100)
    
    surrogate = KernelExpLiteGaussian(sigma=10, lmbda=0.001, D=D, N=N)
    surrogate.fit(Z)
    
    instance = OracleKernelAdaptiveLangevin(D, target_log_pdf, N, surrogate, step_size, gamma2, schedule, acc_star)
    
    return instance

def get_KernelAdaptiveLangevin_instance(D, target_log_pdf, grad):
    step_size = 1.
    schedule = one_over_4th_root_t_schedule
    acc_star = 0.574
    gamma2 = 0.1
    n = 500
    
    surrogate = KernelExpLiteGaussian(sigma=10, lmbda=0.001, D=D, N=n)
    instance = KernelAdaptiveLangevin(D, target_log_pdf, n, surrogate, step_size, gamma2, schedule, acc_star)
    
    return instance


if __name__ == '__main__':
    Log.set_loglevel(20)
    D = 2
    
    bananicity = 0.03
    V = 100
    Z = sample_banana(700, D, bananicity, V)
    target_log_pdf = lambda x: log_banana_pdf(x, bananicity, V, compute_grad=False)
    target_grad = lambda x: log_banana_pdf(x, bananicity, V, compute_grad=True)

    samplers = [
                get_StaticMetropolis_instance(D, target_log_pdf),
                get_AdaptiveMetropolis_instance(D, target_log_pdf),
                get_OracleKameleon_instance(D, target_log_pdf),
                get_Kameleon_instance(D, target_log_pdf),
                get_StaticLangevin_instance(D, target_log_pdf, target_grad),
                get_AdaptiveLangevin_instance(D, target_log_pdf, target_grad),
                get_OracleKernelAdaptiveLangevin_instance(D, target_log_pdf, target_grad),
                get_KernelAdaptiveLangevin_instance(D, target_log_pdf, target_grad),
                ]
    
    for sampler in samplers:
        start = np.zeros(D)
        num_iter = 1000
        
        samples, log_target_densities, times = mini_pmc(sampler, start, num_iter, 10, D)

        visualize_scatter(samples)
        plt.suptitle("%s" % \
                     (sampler.__class__.__name__,))
    plt.show()

