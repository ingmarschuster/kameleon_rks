import os
from kameleon_rks.experiments.tools import store_samples

result_fname = os.path.splitext(os.path.basename(__file__))[0] + ".txt"

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
        
#         # run 1 (initial parameters from Ingmar's pilot runs)
#         proposal_mu = np.array([ -0.17973298, 0.11796741, 0.86733172, 0.52834129, -3.32354247])
#         proposal_var = np.array([  0.01932092, 0.02373668, 0.02096583, 0.1566503 , 0.46316933])
#         proposal_L_C = np.diag(np.sqrt(proposal_var))
#         proposal_L_C *= 2
#         # resuts:
#         # mean: array([ 0.08820961, -0.03436691,  0.7178945,   0.54124475, -3.77499049])
#         # var: array([ 0.01514925,  0.01407534,  0.056966,    0.37502436,  0.5887545])
#         # np.mean(var): 0.201
#         # np.linalg.norm(mean): 3.882
#         # ESS: not printed

#         # run 2 (use parameters of run 1, inflate variance by 4)
#         proposal_mu = np.array([ 0.08820961, -0.03436691, 0.7178945, 0.54124475, -3.77499049])
#         proposal_var = np.array([ 0.01514925, 0.01407534, 0.056966, 0.37502436,  0.5887545])
#         proposal_L_C = np.diag(np.sqrt(proposal_var))
#         proposal_L_C *= 2
#         # results:
#         # mean: array([ 0.13025213, -0.07923937, 0.48663143, 0.17922614, -3.6706272])
#         # var: array([ 0.00990741, 0.01101477, 0.1076653 , 0.6019245 , 0.58744003])
#         # np.mean(var): 0.264
#         # np.linalg.norm(mean): 3.710
#         # ESS: 2.702
        
#         # run 3 (use parameters of run 2, inflate variance by 4)
#         proposal_mu = np.array([ 0.13025213, -0.07923937, 0.48663143, 0.17922614, -3.6706272])
#         proposal_var = np.array([ 0.00990741, 0.01101477, 0.1076653 , 0.6019245 , 0.58744003])
#         proposal_L_C = np.diag(np.sqrt(proposal_var))
#         proposal_L_C *= 2
#         # results:
#         # mean: array([ 0.14816278, -0.10084356,  0.34308655, -0.19531029, -3.58675556])
#         # var: array([ 0.01056904,  0.0128013 ,  0.10549478,  0.56121843,  0.50030183])
#         # np.mean(var): 0.238
#         # np.linalg.norm(mean): 3.613
#         # ESS: 5.031
        
#         # run 4 (use parameters of run 3, inflate variance by 25 to scan for other modes)
#         proposal_mu = np.array([ 0.14816278, -0.10084356, 0.34308655, -0.19531029, -3.58675556])
#         proposal_var = np.array([ 0.01056904, 0.0128013 , 0.10549478, 0.56121843, 0.50030183])
#         proposal_L_C = np.diag(np.sqrt(proposal_var))
#         proposal_L_C *= 5
#         # mean: array([ 0.19404095, -0.14017837,  0.35465807, -0.22049461, -4.53669311])
#         # var: array([ 0.0898059 ,  0.07987   ,  0.60627872,  2.79810551,  4.48372874])
#         # np.mean(var): 1.612
#         # np.linalg.norm(mean): 4.562
#         # ESS: 1.009
        
        # run 5 (use parameters of run 4, inflate variance by 4 for final check)
        # previous run increased proposal variance a lot, and only last 2 components had increased variance
        # pairwise marginal plot however did not reveal multiple modes, so I assume there are not more modes
        proposal_mu = np.array([ 0.19404095, -0.14017837,  0.35465807, -0.22049461, -4.53669311])
        proposal_var = np.array([ 0.0898059 ,  0.07987   ,  0.60627872,  2.79810551,  4.48372874])
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
        
    if False:
        import matplotlib.pyplot as plt
        visualise_pairwise_marginals(samples)
        plt.show()
