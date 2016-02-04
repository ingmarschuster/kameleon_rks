import os

from kameleon_rks.examples.plotting import visualise_pairwise_marginals
from kameleon_rks.experiments.tools import assert_file_has_sha1sum
from kameleon_rks.proposals.Langevin import StaticLangevin


result_fname = os.path.join(os.path.expanduser('~'), "kameleon_rks_results", "kernel_gradient_is", os.path.splitext(os.path.basename(__file__))[0] + ".txt")

if __name__ == '__main__':
    import time
    
    from kameleon_rks.experiments.tools import store_results
    from kameleon_rks.proposals.Metropolis import StaticMetropolis
    from kameleon_rks.proposals.Metropolis import AdaptiveMetropolis
    from kameleon_rks.proposals.Metropolis import AdaptiveIndependentMetropolis
    from kameleon_rks.proposals.Langevin import OracleKernelAdaptiveLangevin, \
                                    KernelAdaptiveLangevin
    from kameleon_rks.samplers.mini_pmc import mini_pmc
    from kameleon_rks.tools.log import Log
    from kernel_exp_family.estimators.finite.gaussian import KernelExpFiniteGaussian
    import numpy as np
    
    from smc2.sv_models import SVoneSP500Model
    
    logger = Log.get_logger()


    def get_StaticMetropolis_instance(D, target_log_pdf, step_size):
        instance = StaticMetropolis(D, target_log_pdf, step_size)
        
        return instance

    def get_AdaptiveMetropolis_instance(D, target_log_pdf, step_size):
        gamma2 = 0.1
        instance = AdaptiveMetropolis(D, target_log_pdf, step_size, gamma2)
        
        return instance

    def get_AdaptiveIndependentMetropolis_instance(D, target_log_pdf, step_size):
        gamma2 = 0.1
        proposal_mu = true_mean
        proposal_L_C = np.linalg.cholesky(true_cov * 2)
        
        instance = AdaptiveIndependentMetropolis(D, target_log_pdf, step_size, gamma2, proposal_mu, proposal_L_C)
        
        return instance

    def get_OracleKernelAdaptiveLangevin_instance(D, target_log_pdf, step_size):
        m = 1000
        Z = benchmark_samples
        
        sigma = 1.3
        lmbda = 1.
        
        surrogate = KernelExpFiniteGaussian(sigma=sigma, lmbda=lmbda, m=m, D=D)

        logger.info("Fitting kernel exp family in batch mode")
        surrogate.fit(Z)
        
        instance = OracleKernelAdaptiveLangevin(D, target_log_pdf, surrogate, step_size)
        
        return instance
    
    def get_KernelAdaptiveLangevin_instance(D, target_log_pdf, step_size):
        m = 1000
        
        sigma = 1.3
        lmbda = 1.
        
        surrogate = KernelExpFiniteGaussian(sigma=sigma, lmbda=lmbda, m=m, D=D)
        
        instance = KernelAdaptiveLangevin(D, target_log_pdf, surrogate, step_size)

        return instance

if __name__ == '__main__':
    Log.set_loglevel(20)
    
    # load benchmark samples, make sure its a particular file version
    benchmark_samples_fname = "pmc_sv_benchmark_samples.txt"
    benchmark_samples_sha1 = "d53e505730c41fbe413188530916d9a402e21a87"
    assert_file_has_sha1sum(benchmark_samples_fname, benchmark_samples_sha1)
    
    benchmark_samples = np.loadtxt(benchmark_samples_fname)
    benchmark_samples = benchmark_samples[np.arange(0, len(benchmark_samples), step=50)]
    true_mean = np.mean(benchmark_samples, axis=0)
    true_var = np.var(benchmark_samples, axis=0)
    true_cov = np.cov(benchmark_samples.T)
    
    num_iter_per_particle = 100
    population_sizes = [50, 20, 30, 10, 40, 5, 100]
    step_sizes = [1, 2, 0.5, 0.1]
    num_repetitions = 30
    
    for _ in range(num_repetitions):
        mdl = SVoneSP500Model()
        
        # number of particles used for integrating out the latent variables
        mdl.mdl_param.NX = mdl.mdl_param.NX * 1
        D = mdl.dim
        
        target_log_pdf = mdl.get_logpdf_closure()
        
        for step_size in step_sizes:
            for population_size in population_sizes:
                num_iter = population_size * num_iter_per_particle
    
                samplers = [
                                get_StaticMetropolis_instance(D, target_log_pdf, step_size),
                                get_AdaptiveMetropolis_instance(D, target_log_pdf, step_size),
                                get_AdaptiveIndependentMetropolis_instance(D, target_log_pdf, step_size),
                                get_OracleKernelAdaptiveLangevin_instance(D, target_log_pdf, step_size),
                                get_KernelAdaptiveLangevin_instance(D, target_log_pdf, step_size),
                            ]
                    
                for sampler in samplers:
                    try:
                        logger.info("%s uses %s" % (sampler.get_name(), dict(sampler.get_parameters())))
                        
                        start = np.array(true_mean)
                        
                        start_time = time.time()
                        samples, log_target_densities, times = mini_pmc(sampler, start, num_iter, population_size)
                        time_taken = time.time() - start_time
        
                        rmse_mean = np.mean((true_mean - np.mean(samples, 0)) ** 2)
                        rmse_var = np.mean((true_var - np.var(samples, 0)) ** 2)
                        rmse_cov = np.mean((true_cov - np.cov(samples.T)) ** 2)
                        
                        logger.info("Storing results under %s" % result_fname)
                        store_results(result_fname,
                                      sampler_name=sampler.get_name(),
                                      D=D,
                                      population_size=population_size,
                                      num_iter_per_particle=num_iter_per_particle,
                                        
                                      rmse_mean=rmse_mean,
                                      rmse_var=rmse_var,
                                      rmse_cov=rmse_cov,
                                      time_taken=time_taken,
                                      
                                      **sampler.get_parameters()
                                      )
                
                        if False:
                            import matplotlib.pyplot as plt
                            visualise_pairwise_marginals(samples)
                            plt.title("%s" % sampler.get_name())
                            
                            if isinstance(sampler, StaticLangevin):
                                plt.figure()
                                plt.grid(True)
                                plt.title("Drift norms %s" % sampler.get_name())
                                plt.hist(sampler.forward_drift_norms)
                            
                            plt.show()
                    except Exception as e:
                        logger.error("Error happened:\n%s\nContinuing with next sampler" % str(e))
