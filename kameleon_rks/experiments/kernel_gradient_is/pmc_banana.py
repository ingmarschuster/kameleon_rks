import os

from kameleon_rks.proposals.Metropolis import AdaptiveIndependentMetropolis


result_fname = os.path.join(os.path.expanduser('~'), "kameleon_rks_results", "kernel_gradient_is", os.path.splitext(os.path.basename(__file__))[0] + ".txt")

if __name__ == "__main__":
    
    import time
    
    from kameleon_rks.densities.banana import log_banana_pdf, sample_banana
    from kameleon_rks.examples.plotting import visualize_scatter_2d, visualise_fit_2d
    from kameleon_rks.experiments.tools import store_results
    from kameleon_rks.proposals.Kameleon import gamma_median_heuristic
    from kameleon_rks.proposals.Langevin import AdaptiveLangevin
    from kameleon_rks.proposals.Langevin import StaticLangevin
    from kameleon_rks.proposals.Metropolis import StaticMetropolis
    from kameleon_rks.proposals.Metropolis import AdaptiveMetropolis
    from kameleon_rks.proposals.Langevin import OracleKernelAdaptiveLangevin, \
                                    KernelAdaptiveLangevin
    from kameleon_rks.samplers.mini_pmc import mini_pmc
    from kameleon_rks.tools.convergence_stats import mmd_to_benchmark_sample
    from kameleon_rks.tools.log import Log
    from kernel_exp_family.estimators.finite.gaussian import KernelExpFiniteGaussian
    from kernel_exp_family.estimators.parameter_search_bo import BayesOptSearch
    import numpy as np
    
    logger = Log.get_logger()

    def one_over_sqrt_t_schedule(t):
        return 1. / np.sqrt(1 + t)
    
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
    
    def get_StaticLangevin_instance(D, target_log_pdf, grad):
        step_size = 1.
        instance = StaticLangevin(D, target_log_pdf, grad, step_size)
        
        return instance
    
    def get_AdaptiveLangevin_instance(D, target_log_pdf, grad):
        step_size = 1.
        instance = AdaptiveLangevin(D, target_log_pdf, grad, step_size)
        
        return instance
    
    def get_OracleKernelAdaptiveLangevin_instance(D, target_log_pdf, grad):
        step_size = 1.
        m = 500
        N = 5000
        Z = sample_banana(N, D, bananicity, V)
        
        surrogate = KernelExpFiniteGaussian(sigma=10, lmbda=.001, m=m, D=D)
        surrogate.fit(Z)
        
        if False:
            param_bounds = {'sigma': [-2, 3]}
            bo = BayesOptSearch(surrogate, Z, param_bounds)
            best_params = bo.optimize()
            surrogate.set_parameters_from_dict(best_params)
        
        if False:
            sigma = 1. / gamma_median_heuristic(Z)
            surrogate.set_parameters_from_dict({'sigma': sigma})
        
        logger.info("kernel exp family uses %s" % surrogate.get_parameters())
    
        if False:
            import matplotlib.pyplot as plt
            Xs = np.linspace(-30, 30, 50)
            Ys = np.linspace(-20, 40, 50)
            visualise_fit_2d(surrogate, Z, Xs, Ys)
            plt.show()
            
        instance = OracleKernelAdaptiveLangevin(D, target_log_pdf, surrogate, step_size)
        
        return instance
    
    def get_KernelAdaptiveLangevin_instance(D, target_log_pdf, grad):
        step_size = 1.
        m = 500
        
        surrogate = KernelExpFiniteGaussian(sigma=10, lmbda=1., m=m, D=D)
        logger.info("kernel exp family uses %s" % surrogate.get_parameters())
        
        instance = KernelAdaptiveLangevin(D, target_log_pdf, surrogate, step_size)

        return instance
    
if __name__ == '__main__':
    Log.set_loglevel(20)
    
    D = 2
    
    bananicity = 0.03
    V = 100
    true_mean = np.zeros(D)
    true_var = np.ones(D)
    true_var[0] = 100
    true_var[1] = 20
    true_cov = np.diag(true_var)
    
    num_iter_per_particle = 200
    population_sizes = [5, 10, 25, 50, 100]
    
    num_repetitions = 30
    
    num_benchmark_samples = 1000
    rng_state = np.random.get_state()
    np.random.seed(0)
    benchmark_sample = sample_banana(num_benchmark_samples, D, bananicity, V)
    np.random.set_state(rng_state)
    
    for _ in range(num_repetitions):
        for population_size in population_sizes:
            num_iter = population_size * num_iter_per_particle
            start = np.zeros(D)
            
            target_log_pdf = lambda x: log_banana_pdf(x, bananicity, V, compute_grad=False)
            target_grad = lambda x: log_banana_pdf(x, bananicity, V, compute_grad=True)

            samplers = [
                            get_StaticMetropolis_instance(D, target_log_pdf),
                            get_AdaptiveMetropolis_instance(D, target_log_pdf),
                            get_AdaptiveIndependentMetropolis_instance(D, target_log_pdf),
                            get_StaticLangevin_instance(D, target_log_pdf, target_grad),
                            get_AdaptiveLangevin_instance(D, target_log_pdf, target_grad),
                            get_OracleKernelAdaptiveLangevin_instance(D, target_log_pdf, target_grad),
                            get_KernelAdaptiveLangevin_instance(D, target_log_pdf, target_grad),
                        ]
                
            for sampler in samplers:
                start_time = time.time()
                samples, log_target_densities, times = mini_pmc(sampler, start, num_iter, population_size)
                time_taken = time.time() - start_time

                mmd = mmd_to_benchmark_sample(samples, benchmark_sample, degree=3)
                rmse_mean = np.mean((true_mean - np.mean(samples, 0)) ** 2)
                rmse_cov = np.mean((true_cov - np.cov(samples.T)) ** 2)
                
                logger.info("Storing results under %s" % result_fname)
                store_results(result_fname,
                              sampler_name=sampler.get_name(),
                              D=D,
                              bananicity=bananicity,
                              V=V,
                              num_benchmark_samples=num_benchmark_samples,
                              population_size=population_size,
                              num_iter_per_particle=num_iter_per_particle,
                                
                              mmd=mmd,
                              rmse_mean=rmse_mean,
                              rmse_cov=rmse_cov,
                              time_taken=time_taken,
                              )
        
                if False:
                    import matplotlib.pyplot as plt
                    visualize_scatter_2d(samples)
                    plt.title("%s" % sampler.get_name())
                    
                    if isinstance(sampler, OracleKernelAdaptiveLangevin):
                        Xs = np.linspace(-30, 30, 50)
                        Ys = np.linspace(-20, 40, 50)
                        visualise_fit_2d(sampler.surrogate, samples, Xs, Ys)
                    
                    if isinstance(sampler, StaticLangevin):
                        plt.figure()
                        plt.grid(True)
                        plt.title("Drift norms %s" % sampler.get_name())
                        plt.hist(sampler.forward_drift_norms)
                    
                    plt.show()
