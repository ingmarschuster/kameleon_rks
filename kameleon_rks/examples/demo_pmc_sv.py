from kameleon_rks.densities.banana import sample_banana, log_banana_pdf
from kameleon_rks.examples.plotting import visualize_scatter_2d
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

from smc2.sv_models import SVoneSP500Model


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
    mdl = SVoneSP500Model()
    
    #the following is the number of particles used for integrating out the latent variables
    mdl.mdl_param.NX = mdl.mdl_param.NX * 1
    
    D = mdl.dim
    pop_size=50
    target_log_pdf = mdl.get_logpdf_closure()

    samplers = [
#                get_StaticMetropolis_instance(D, target_log_pdf),
#                get_AdaptiveMetropolis_instance(D, target_log_pdf),
#                get_OracleKameleon_instance(D, target_log_pdf),
#                get_Kameleon_instance(D, target_log_pdf),
#                get_StaticLangevin_instance(D, target_log_pdf, target_grad),
#                get_AM_5(D, target_log_pdf),
                get_AM_1(D, target_log_pdf),
#                get_AM_2(D, target_log_pdf),
#                get_AM_05(D, target_log_pdf),
                ]
    
    for sampler in samplers:
        print(sampler.__class__)
        start = mdl.rvs(pop_size)
        num_iter = 5000
        
        samples, log_target_densities, times = mini_pmc(sampler, start, num_iter, pop_size)
        mom_samp = np.array([(samples**i).mean(0) for i in range(1,4)])
        
        visualize_scatter_2d(samples[:,:2])
        Log.get_logger().info('===='+str(sampler.step_size)+'====')

    plt.show()

