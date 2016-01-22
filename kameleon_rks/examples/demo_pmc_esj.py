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
from kameleon_rks.tools.esj import esj
from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
import matplotlib.pyplot as plt
import numpy as np


def one_over_4th_root_t_schedule(t):
    return 1. / np.sqrt(1 + t)



def get_AM_05(D, target_log_pdf):
    
    step_size = 0.5
    acc_star = 0.234
    gamma2 = 0.1
    instance = AdaptiveMetropolis(D, target_log_pdf, step_size, gamma2, one_over_4th_root_t_schedule, acc_star)
    
    return instance


def get_AM_1(D, target_log_pdf):
    
    step_size = 1.
    acc_star = 0.234
    gamma2 = 0.1
    instance = AdaptiveMetropolis(D, target_log_pdf, step_size, gamma2, one_over_4th_root_t_schedule, acc_star)
    
    return instance
    
def get_AM_2(D, target_log_pdf):
    
    step_size = 2.
    acc_star = 0.234
    gamma2 = 0.1
    instance = AdaptiveMetropolis(D, target_log_pdf, step_size, gamma2, one_over_4th_root_t_schedule, acc_star)
    
    return instance

def get_AM_5(D, target_log_pdf):
    
    step_size = 5.
    acc_star = 0.234
    gamma2 = 0.1
    instance = AdaptiveMetropolis(D, target_log_pdf, step_size, gamma2, one_over_4th_root_t_schedule, acc_star)
    
    return instance
    


if __name__ == '__main__':
    Log.set_loglevel(20)
    D = 2
    pop_size=10
    bananicity = 0.03
    V = 100
    Z = sample_banana(700, D, bananicity, V)
    moments = np.array([(Z**i).mean(0) for i in range(1,4)])
    target_log_pdf = lambda x: log_banana_pdf(x, bananicity, V, compute_grad=False)
    target_grad = lambda x: log_banana_pdf(x, bananicity, V, compute_grad=True)

    samplers = [
#                get_StaticMetropolis_instance(D, target_log_pdf),
#                get_AdaptiveMetropolis_instance(D, target_log_pdf),
#                get_OracleKameleon_instance(D, target_log_pdf),
#                get_Kameleon_instance(D, target_log_pdf),
#                get_StaticLangevin_instance(D, target_log_pdf, target_grad),
                get_AM_5(D, target_log_pdf),
                get_AM_1(D, target_log_pdf),
                get_AM_2(D, target_log_pdf),
                get_AM_05(D, target_log_pdf),
                ]
    
    for sampler in samplers:
        start = np.zeros(D)
        num_iter = 1000
        
        samples, log_target_densities, times = mini_pmc(sampler, start, num_iter, pop_size, D)
        mom_samp = np.array([(samples**i).mean(0) for i in range(1,4)])
        
        visualize_scatter(samples)
        Log.get_logger().info('===='+str(sampler.step_size)+'====')
        #the following two should be 0 ideally
        Log.get_logger().info(np.mean((esj(samples,pop_size)-1)**2)) 
        Log.get_logger().info(np.mean(((mom_samp / moments)-1)**2))
        plt.suptitle("%s" % \
                     (sampler.__class__.__name__,))
    plt.show()

