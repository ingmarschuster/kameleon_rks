from kameleon_rks.examples.plotting import visualise_fit_2d
from kameleon_rks.experiments.tools import assert_file_has_sha1sum
from kernel_exp_family.estimators.finite.gaussian import KernelExpFiniteGaussian
import matplotlib.pyplot as plt
import numpy as np


class empty_class():
    pass

if __name__ == "__main__":
        
    # load benchmark samples, make sure its a particular file version
    benchmark_samples_fname = "pmc_sv_benchmark_samples.txt"
    benchmark_samples_sha1 = "d53e505730c41fbe413188530916d9a402e21a87"
    assert_file_has_sha1sum(benchmark_samples_fname, benchmark_samples_sha1)
    
    benchmark_samples = np.loadtxt(benchmark_samples_fname)
    benchmark_samples = benchmark_samples[np.arange(0, len(benchmark_samples), step=50)]
    
    true_mean = np.mean(benchmark_samples, axis=0)
    
    m = 1000
    Z = benchmark_samples
    
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
        surrogate = KernelExpFiniteGaussian(sigma=sigma, lmbda=lmbda, m=m, D=Z.shape[1])
        vals = surrogate.xvalidate_objective(Z)
        Js_mean[i] = np.mean(vals)
        Js_var[i] = np.var(vals)
        print "log2_sigma: %.3f, sigma: %.3f, mean: %.3f, var: %.3f" % (log2_sigma, sigma, Js_mean[i], Js_var[i])
        
        surrogate = KernelExpFiniteGaussian(sigma=sigma, lmbda=lmbda, m=m, D=Z.shape[1])
        surrogate.fit(benchmark_samples)
        fake = empty_class()
        fake.log_pdf = lambda x_2d: surrogate.log_pdf(np.hstack((x_2d, true_mean[2:])))
        fake.grad = lambda x_2d: surrogate.grad(np.hstack((x_2d, true_mean[2:])))
                                                                       
        visualise_fit_2d(fake, benchmark_samples,
                         Xs=np.linspace(benchmark_samples[:, 0].min(), benchmark_samples[:, 0].max(), 30),
                         Ys=np.linspace(benchmark_samples[:, 1].min(), benchmark_samples[:, 1].max(), 30),
                         
                         )
        plt.show()

    plt.plot(log2_sigmas, Js_mean, 'b-')
    plt.plot(log2_sigmas, Js_mean - 2 * np.sqrt(Js_var[i]), 'b--')
    plt.plot(log2_sigmas, Js_mean + 2 * np.sqrt(Js_var[i]), 'b--')
    plt.show()
    
