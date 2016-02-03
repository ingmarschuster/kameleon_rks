import os

from kameleon_rks.examples.plotting import visualise_pairwise_marginals
from kameleon_rks.experiments.tools import assert_file_has_sha1sum
from kameleon_rks.proposals.Kameleon import gamma_median_heuristic
from kameleon_rks.proposals.Langevin import StaticLangevin
from kernel_exp_family.estimators.parameter_search_bo import plot_bayesopt_model_1d


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
    from kernel_exp_family.estimators.parameter_search_bo import BayesOptSearch
    import numpy as np
    
    from smc2.sv_models import SVoneSP500Model
    
    logger = Log.get_logger()


    def get_StaticMetropolis_instance(D, target_log_pdf):
        step_size = 1.
        instance = StaticMetropolis(D, target_log_pdf, step_size)
        
        return instance

    def get_AdaptiveMetropolis_instance(D, target_log_pdf):
        step_size = 1.
        gamma2 = 0.1
        instance = AdaptiveMetropolis(D, target_log_pdf, step_size, gamma2)
        
        return instance

    def get_AdaptiveIndependentMetropolis_instance(D, target_log_pdf):
        gamma2 = 0.1
        proposal_mu = true_mean
        proposal_L_C = np.linalg.cholesky(true_cov * 2)
        
        instance = AdaptiveIndependentMetropolis(D, target_log_pdf, gamma2, proposal_mu, proposal_L_C)
        
        return instance

    def get_OracleKernelAdaptiveLangevin_instance(D, target_log_pdf):
        step_size = 1.
        m = 1500
        Z = benchmark_samples
        
        sigma = np.exp(-0.7)
        lmbda = 1.
        
        surrogate = KernelExpFiniteGaussian(sigma=sigma, lmbda=lmbda, m=m, D=D)
        
        if False:
            import matplotlib.pyplot as plt
            param_bounds = {'sigma': [-2, 2]}
            bo = BayesOptSearch(surrogate, Z, param_bounds, num_initial_evaluations=15,
                                objective_log=True, objective_log_bound=0)
            best_params = bo.optimize(num_iter=15)
            surrogate.set_parameters_from_dict(best_params)
            plot_bayesopt_model_1d(bo)
            plt.show()
        
        if False:
            sigma = 1. / gamma_median_heuristic(Z)
            surrogate.set_parameters_from_dict({'sigma': sigma})
        
        logger.info("kernel exp family uses %s" % surrogate.get_parameters())
        surrogate.fit(Z)
    
        instance = OracleKernelAdaptiveLangevin(D, target_log_pdf, surrogate, step_size)
        
        return instance
    
    def get_KernelAdaptiveLangevin_instance(D, target_log_pdf):
        step_size = 1.
        m = 1000
        
        sigma = np.exp(-0.7)
        lmbda = 1.
        
        surrogate = KernelExpFiniteGaussian(sigma=sigma, lmbda=lmbda, m=m, D=D)
        logger.info("kernel exp family uses %s" % surrogate.get_parameters())
        
        instance = KernelAdaptiveLangevin(D, target_log_pdf, surrogate, step_size)

        return instance

if __name__ == '__main__':
    Log.set_loglevel(20)
    
    # load benchmark samples, make sure its a particular file version
    benchmark_samples_fname = "pmc_sv_benchmark_samples.txt"
    benchmark_samples_sha1 = "ffa274673a0388c5eccf4210d34c2281ad9e3157"
    assert_file_has_sha1sum(benchmark_samples_fname, benchmark_samples_sha1)
    
    benchmark_samples = np.loadtxt(benchmark_samples_fname)
    true_mean = np.mean(benchmark_samples, axis=0)
    true_var = np.var(benchmark_samples, axis=0)
    true_cov = np.cov(benchmark_samples.T)
    
    num_iter_per_particle = 200
    population_sizes = [10, 20, 50, 100]

    num_repetitions = 30
    
    for _ in range(num_repetitions):
        mdl = SVoneSP500Model()
        
        # number of particles used for integrating out the latent variables
        mdl.mdl_param.NX = mdl.mdl_param.NX * 1
        D = mdl.dim
        
        target_log_pdf = mdl.get_logpdf_closure()
        
        for population_size in population_sizes:
            num_iter = population_size * num_iter_per_particle

            samplers = [
                            get_StaticMetropolis_instance(D, target_log_pdf),
                            get_AdaptiveMetropolis_instance(D, target_log_pdf),
                            get_AdaptiveIndependentMetropolis_instance(D, target_log_pdf),
                            get_OracleKernelAdaptiveLangevin_instance(D, target_log_pdf),
                            get_KernelAdaptiveLangevin_instance(D, target_log_pdf),
                        ]
                
            for sampler in samplers:
                try:
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
