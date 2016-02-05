from kameleon_rks.examples.plotting import visualise_fit_2d, \
    visualise_pairwise_marginals
from kameleon_rks.experiments.tools import assert_file_has_sha1sum
from kameleon_rks.proposals.Kameleon import gamma_median_heuristic
from kernel_exp_family.estimators.finite.gaussian import KernelExpFiniteGaussian
import matplotlib.pyplot as plt
import numpy as np


class empty_class():
    pass

def get_benchmark_samples_pmc():
    # load benchmark samples, make sure its a particular file version
    benchmark_samples_fname = "pmc_sv_benchmark_samples.txt"
    benchmark_samples_sha1 = "d53e505730c41fbe413188530916d9a402e21a87"
    assert_file_has_sha1sum(benchmark_samples_fname, benchmark_samples_sha1)
    
    benchmark_samples = np.loadtxt(benchmark_samples_fname)
    benchmark_samples = benchmark_samples[np.arange(0, len(benchmark_samples), step=50)]
    return benchmark_samples

def get_benchmark_samples_mcmc():
    # load benchmark samples, make sure its a particular file version
    benchmark_samples_fname = "mcmc_sv_benchmark_samples.txt"
    benchmark_samples_sha1 = "dd71899bf8ead3972de45543b09af95dc858a208"
    
    assert_file_has_sha1sum(benchmark_samples_fname, benchmark_samples_sha1)
    
    benchmark_samples = np.loadtxt(benchmark_samples_fname)
    benchmark_samples = benchmark_samples[np.arange(0, len(benchmark_samples), step=200)]
    return benchmark_samples

if __name__ == "__main__":

    benchmark_samples = get_benchmark_samples_mcmc()
    true_mean = np.mean(benchmark_samples, axis=0)
    true_var = np.var(benchmark_samples, axis=0)
    print("%d benchmark samples" % len(benchmark_samples))
    print "mean:", repr(true_mean)
    print "var:", repr(true_var)
    print "np.mean(var): %.3f" % np.mean(true_var)
    print "np.linalg.norm(mean): %.3f" % np.linalg.norm(true_mean)
    print "median heuristic sigma: %.3f" % (1. / gamma_median_heuristic(benchmark_samples))
    
    visualise_pairwise_marginals(benchmark_samples)
    plt.show()
    
    m = 1000
    
    lmbda = 1.
    res = 10
    log2_sigmas = np.linspace(-4, 10, res)
    log2_sigmas = np.log2([0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
    
    print "log2_sigmas:", log2_sigmas
    print "sigmas:", 2 ** log2_sigmas
    Js_mean = np.zeros(res)
    Js_var = np.zeros(res)
    
    for i, log2_sigma in enumerate(log2_sigmas):
        sigma = 2 ** log2_sigma
        surrogate = KernelExpFiniteGaussian(sigma=sigma, lmbda=lmbda, m=m, D=benchmark_samples.shape[1])
        vals = surrogate.xvalidate_objective(benchmark_samples)
        Js_mean[i] = np.mean(vals)
        Js_var[i] = np.var(vals)
        print "log2_sigma: %.3f, sigma: %.3f, mean: %.3f, var: %.3f" % (log2_sigma, sigma, Js_mean[i], Js_var[i])
        
        surrogate = KernelExpFiniteGaussian(sigma=sigma, lmbda=lmbda, m=m, D=benchmark_samples.shape[1])
        surrogate.fit(benchmark_samples)
        fake = empty_class()
        
        def replace_2(x_2d, a, i, j):
            a = a.copy()
            a[i] = x_2d[0]
            a[j] = x_2d[1]
            return a
            
        for i in range(benchmark_samples.shape[1]):
            for j in range(benchmark_samples.shape[1]):
                if i == j:
                    continue
                fake.log_pdf = lambda x_2d: surrogate.log_pdf(replace_2(x_2d, true_mean, i, j))
                fake.grad = lambda x_2d: surrogate.grad(replace_2(x_2d, true_mean, i, j))
                                                                               
                visualise_fit_2d(fake, benchmark_samples[:, [i, j]],
                                 Xs=np.linspace(benchmark_samples[:, i].min(), benchmark_samples[:, i].max(), 30),
                                 Ys=np.linspace(benchmark_samples[:, j].min(), benchmark_samples[:, j].max(), 30),
                                 
                                 )
                plt.show()

    plt.plot(log2_sigmas, Js_mean, 'b-')
    plt.plot(log2_sigmas, Js_mean - 2 * np.sqrt(Js_var[i]), 'b--')
    plt.plot(log2_sigmas, Js_mean + 2 * np.sqrt(Js_var[i]), 'b--')
    plt.show()
    
