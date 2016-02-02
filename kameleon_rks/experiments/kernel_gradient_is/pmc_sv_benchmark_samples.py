import os
from kameleon_rks.experiments.tools import store_samples

result_fname = os.path.join(os.path.expanduser('~'), "kameleon_rks_results", "kernel_gradient_is", os.path.splitext(os.path.basename(__file__))[0] + ".txt")

if __name__ == '__main__':
    import time
    
    from kameleon_rks.examples.plotting import visualise_pairwise_marginals
    from kameleon_rks.proposals.Metropolis import AdaptiveIndependentMetropolis
    from kameleon_rks.samplers.mini_pmc import mini_pmc
    from kameleon_rks.tools.log import Log
    import numpy as np
    
    from smc2.sv_models import SVoneSP500Model
    
    logger = Log.get_logger()


    def get_AdaptiveIndependentMetropolis_instance(D, target_log_pdf):
        gamma2 = 0.1
        
        # these parameters have been estimated from previous runs
        proposal_mu = np.array([-0.17973298, 0.11796741, 0.86733172, 0.52834129, -3.32354247])
        proposal_var = np.array([ 0.01932092, 0.02373668, 0.02096583, 0.1566503 , 0.46316933])
        proposal_L_C = np.diag(np.sqrt(proposal_var))
        
        # larger support for proposal
        proposal_L_C *= 2
        
        instance = AdaptiveIndependentMetropolis(D, target_log_pdf, gamma2, proposal_mu, proposal_L_C)
        
        return instance

if __name__ == '__main__':
    Log.set_loglevel(10)
    
    num_iter_per_particle = 100
    population_size = 50
    num_iter = population_size * num_iter_per_particle
    
    mdl = SVoneSP500Model()
    target_log_pdf = mdl.get_logpdf_closure()

    # number of particles used for integrating out the latent variables
    mdl.mdl_param.NX = mdl.mdl_param.NX * 1
    
    D = mdl.dim
    start = np.zeros(D)

    sampler = get_AdaptiveIndependentMetropolis_instance(D, target_log_pdf)
        
    start_time = time.time()
    samples, log_target_densities, times = mini_pmc(sampler, start, num_iter, population_size)
    time_taken = time.time() - start_time

    logger.info("Storing results under %s" % result_fname)
    store_samples(samples, result_fname)
    
    print "mean:\n", np.mean(samples, axis=0)
    print "var:\n", np.var(samples, axis=0)
        
    if True:
        import matplotlib.pyplot as plt
        visualise_pairwise_marginals(samples)
        plt.show()
