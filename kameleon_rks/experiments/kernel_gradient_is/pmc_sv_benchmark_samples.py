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
        
#         # run 1
#         proposal_mu = np.array([-0.17973298, 0.11796741, 0.86733172, 0.52834129, -3.32354247])
#         proposal_var = np.array([ 0.01932092, 0.02373668, 0.02096583, 0.1566503 , 0.46316933])
#         proposal_L_C = np.diag(np.sqrt(proposal_var))
#         proposal_L_C *= 2
#         # resuts:
#         # mean: [0.08820961, -0.03436691,  0.7178945,   0.54124475, -3.77499049]
#         # var: [0.01514925,  0.01407534,  0.056966,    0.37502436,  0.5887545]
#         # np.mean(var): 0.201
#         # np.linalg.norm(mean): 3.882
#         # ESS: not printed

#         # run 2
#         proposal_mu = np.array([ 0.08820961, -0.03436691, 0.7178945, 0.54124475, -3.77499049 ])
#         proposal_var = np.array([ 0.01514925, 0.01407534, 0.056966, 0.37502436,  0.5887545 ])
#         proposal_L_C = np.diag(np.sqrt(proposal_var))
#         proposal_L_C *= 2
#         # results:
#         # mean: array([ 0.13025213, -0.07923937, 0.48663143, 0.17922614, -3.6706272 ])
#         # var: array([ 0.00990741, 0.01101477, 0.1076653 , 0.6019245 , 0.58744003])
#         # np.mean(var): 0.264
#         # np.linalg.norm(mean): 3.710
#         # ESS: 2.702
        
        # run 3
        proposal_mu = np.array([ 0.13025213, -0.07923937, 0.48663143, 0.17922614, -3.6706272 ])
        proposal_var = np.array([ 0.00990741, 0.01101477, 0.1076653 , 0.6019245 , 0.58744003])
        proposal_L_C = np.diag(np.sqrt(proposal_var))
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
    
    mean = np.mean(samples, axis=0)
    var = np.var(samples, axis=0)
    print "mean:", repr(mean)
    print "var:", repr(var)
    print "np.mean(var): %.3f" % np.mean(var)
    print "np.linalg.norm(mean): %.3f" % np.linalg.norm(mean)
    print "ESS: %.3f" % sampler.get_current_ess()
        
    if True:
        import matplotlib.pyplot as plt
        visualise_pairwise_marginals(samples)
        plt.show()
