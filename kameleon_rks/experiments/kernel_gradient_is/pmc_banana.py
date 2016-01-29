import os
def gen_result_fname():
    return os.path.join(os.path.expanduser('~'), "kameleon_rks_results", "kernel_gradient_is", os.path.splitext(os.path.basename(__file__))[0] + ".txt")

if __name__ == "__main__":
    
    from scipy.misc.common import logsumexp
    import time
    
    from kameleon_rks.densities.banana import log_banana_pdf, sample_banana
    from kameleon_rks.examples.plotting import visualize_scatter, visualise_fit
    from kameleon_rks.experiments.tools import store_results
    from kameleon_rks.proposals.Kameleon import gamma_median_heuristic
    from kameleon_rks.proposals.Langevin import AdaptiveLangevin, \
        OracleKernelAdaptiveLangevin, KernelAdaptiveLangevin
    from kameleon_rks.proposals.Metropolis import AdaptiveMetropolis
    from kameleon_rks.samplers.mini_pmc import mini_pmc
    from kameleon_rks.tools.convergence_stats import mmd_to_benchmark_sample, \
        min_ess
    from kameleon_rks.tools.log import Log
    from kernel_exp_family.estimators.finite.gaussian import KernelExpFiniteGaussian
    from kernel_exp_family.estimators.parameter_search_bo import BayesOptSearch
    import numpy as np
    
    logger = Log.get_logger()

    def one_over_sqrt_t_schedule(t):
        return 1. / np.sqrt(1 + t)
    
    def get_AdaptiveMetropolis_instance(D, target_log_pdf):
        
        step_size = 1.
        schedule = one_over_sqrt_t_schedule
        gamma2 = 0.1
        instance = AdaptiveMetropolis(D, target_log_pdf, step_size, gamma2, schedule)
        
        return instance
    
    def get_AdaptiveLangevin_instance(D, target_log_pdf, grad):
        
        step_size = 1.
        schedule = one_over_sqrt_t_schedule
        
        instance = AdaptiveLangevin(D, target_log_pdf, grad, step_size, schedule)
        
        return instance
    
    def get_OracleKernelAdaptiveLangevin_instance(D, target_log_pdf, grad):
        
        step_size = 1.
        schedule = one_over_sqrt_t_schedule
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
            visualise_fit(surrogate, Z, Xs, Ys)
            plt.show()
            
        instance = OracleKernelAdaptiveLangevin(D, target_log_pdf, surrogate, step_size, schedule)
        
        return instance
    
    def get_KernelAdaptiveLangevin_instance(D, target_log_pdf, grad):
        
        step_size = 1.
        schedule = one_over_sqrt_t_schedule
        m = 100
        
        surrogate = KernelExpFiniteGaussian(sigma=10, lmbda=1., m=m, D=D)
        surrogate.sum_weights = np.exp(logsumexp([target_log_pdf(x) for x in sample_banana(m, D, bananicity, V)]))
        logger.info("kernel exp family uses %s" % surrogate.get_parameters())
        
        # no schedule means no step size adaptation
        instance = KernelAdaptiveLangevin(D, target_log_pdf, surrogate, step_size, schedule=None)
        
        return instance
    
    if __name__ == '__main__':
        Log.set_loglevel(20)
        D = 2
        
        bananicity = 0.1
        V = 100
        true_mean = np.zeros(D)
        true_var = np.ones(D)
        true_var[0] = 100
        true_var[1] = 200
        
        num_population = 200
        start = np.zeros(D)
        num_iter = 2000
        
        rng_state = np.random.get_state()
        np.random.seed(0)
        benchmark_sample = sample_banana(5000, D, bananicity, V)
        np.random.set_state(rng_state)
        
        target_log_pdf = lambda x: log_banana_pdf(x, bananicity, V, compute_grad=False)
        target_grad = lambda x: log_banana_pdf(x, bananicity, V, compute_grad=True)
    
        
        num_repetitions = 30
        for _ in range(num_repetitions):
            samplers = [
                        get_AdaptiveMetropolis_instance(D, target_log_pdf),
                        get_AdaptiveLangevin_instance(D, target_log_pdf, target_grad),
                        get_OracleKernelAdaptiveLangevin_instance(D, target_log_pdf, target_grad),
    #                     get_KernelAdaptiveLangevin_instance(D, target_log_pdf, target_grad),
                        
                        ]
                
            for sampler in samplers:
                start_time = time.time()
                samples, log_target_densities, times = mini_pmc(sampler, start, num_iter, num_population)
                time_taken = time.time() - start_time
                
                mmd = mmd_to_benchmark_sample(samples, benchmark_sample, degree=3)
                ess = min_ess(samples)
                rmse_mean = np.linalg.norm(true_mean - np.mean(samples, 0))
                rmse_var = np.linalg.norm(true_var - np.var(samples, 0))
                norm_of_mean = np.linalg.norm(np.mean(samples, axis=0))
                
                logger.info("Storing results under %s" % gen_result_fname())
                store_results(gen_result_fname(),
                              sampler_name=sampler.get_name(),
                              D=D,
                              bananicity=bananicity,
                              V=V,
                              num_population=num_population,
                              num_iter=num_iter,
                              
                              mmd=mmd,
                              ess=ess,
                              rmse_mean=rmse_mean,
                              rmse_var=rmse_var,
                              norm_of_mean=norm_of_mean,
                              time_taken=time_taken,
                              )
        
                if False:
                    import matplotlib.pyplot as plt
                    visualize_scatter(samples)
                    plt.suptitle("%s" % \
                                 (sampler.get_name()))
                    
                    if isinstance(sampler, OracleKernelAdaptiveLangevin):
                        Xs = np.linspace(-30, 30, 50)
                        Ys = np.linspace(-20, 40, 50)
                        visualise_fit(sampler.surrogate, samples, Xs, Ys)
                  
                    plt.show()
        
