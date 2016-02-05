import os

from kameleon_rks.examples.plotting import visualise_trace_2d
from kameleon_rks.experiments.tools import store_samples
from kameleon_rks.proposals.Metropolis import AdaptiveMetropolis
from kameleon_rks.samplers.mini_mcmc import mini_mcmc
from kameleon_rks.tools.convergence_stats import min_ess


result_fname = os.path.splitext(os.path.basename(__file__))[0] + ".txt"

if __name__ == '__main__':
    from kameleon_rks.examples.plotting import visualise_pairwise_marginals
    from kameleon_rks.tools.log import Log
    import numpy as np
    
    from smc2.sv_models import SVoneSP500Model
    
    logger = Log.get_logger()

    def one_over_sqrt_t_schedule(t):
        return 1. / np.sqrt(1 + t)
    
    def get_AdaptiveMetropolis_instance(D, target_log_pdf):
        gamma2 = 1.
        step_size = 0.01
        acc_star = 0.234
        schedule = one_over_sqrt_t_schedule
        instance = AdaptiveMetropolis(D, target_log_pdf, step_size, gamma2, schedule, acc_star)
        
        return instance

if __name__ == '__main__':
    Log.set_loglevel(20)
    
    num_iter = 5000
    
    mdl = SVoneSP500Model()
    target_log_pdf = mdl.get_logpdf_closure()

    # number of particles used for integrating out the latent variables
    mdl.mdl_param.NX = mdl.mdl_param.NX * 1
    
    D = mdl.dim
    
    # from initial runs
    initial_mean = np.array([ 0.19404095, -0.14017837,  0.35465807, -0.22049461, -4.53669311])
    start = initial_mean

    sampler = get_AdaptiveMetropolis_instance(D, target_log_pdf)
    samples, proposals, accepted, acc_prob, log_pdf, times, step_sizes = mini_mcmc(sampler, start, num_iter, D)

    logger.info("Storing results under %s" % result_fname)
    store_samples(samples, result_fname)
    
    mean = np.mean(samples, axis=0)
    var = np.var(samples, axis=0)
    print "mean:", repr(mean)
    print "var:", repr(var)
    print "np.mean(var): %.3f" % np.mean(var)
    print "np.linalg.norm(mean): %.3f" % np.linalg.norm(mean)
    print "min ESS: %.3f" % min_ess(samples)
        
    if False:
        import matplotlib.pyplot as plt
        visualise_trace_2d(samples, log_pdf, accepted, step_sizes)
        
        plt.figure()
        visualise_pairwise_marginals(samples)
        plt.show()
