from kameleon_rks.densities.banana import log_banana_pdf
from kameleon_rks.examples.plotting import visualise_trace
from kameleon_rks.mcmc.mini_mcmc import mini_mcmc
from kameleon_rks.proposals.Metropolis import StaticMetropolis
from kameleon_rks.tools.log import Log
import matplotlib.pyplot as plt
import numpy as np


def sqrt_schedule(t):
    return 1 / np.sqrt(1 + t)

def get_StaticMetropolisInstance(D, target_log_pdf):
    
    step_size = 8.
    schedule = sqrt_schedule
    acc_star = 0.234
    instance = StaticMetropolis(D, target_log_pdf, step_size, schedule, acc_star)
    
    return instance


if __name__ == '__main__':
    Log.set_loglevel(20)
    D = 2
    
    bananicity = 0.03
    V = 100
    target_log_pdf = lambda x: log_banana_pdf(x, bananicity, V, compute_grad=False)

    samplers = [
                get_StaticMetropolisInstance(D, target_log_pdf),
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
