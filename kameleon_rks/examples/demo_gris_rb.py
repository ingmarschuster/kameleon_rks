from kameleon_rks.densities.banana import sample_banana, log_banana_pdf
from kameleon_rks.examples.plotting import visualize_scatter_2d
from kameleon_rks.proposals.Kameleon import gamma_median_heuristic,\
    OracleKameleon, Kameleon
from kameleon_rks.proposals.Langevin import StaticLangevin, AdaptiveLangevin,\
    OracleKernelAdaptiveLangevin, KernelAdaptiveLangevin
from kameleon_mcmc.kernel.PolynomialKernel import PolynomialKernel
from kameleon_rks.proposals.Langevin import StaticLangevin
from kameleon_rks.samplers.mini_rb_pmc import mini_rb_pmc
from kameleon_rks.samplers.mini_mcmc import mini_mcmc
from kameleon_rks.tools.log import Log
from kameleon_rks.tools.esj import esj
from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
import matplotlib.pyplot as plt
import numpy as np

from numpy import log, exp
from scipy.misc import logsumexp
from kameleon_rks.samplers.tools import system_res

import pylab as pl
import matplotlib.pyplot as plt

def apply_to_mg(func, *mg):
    #apply a function to points on a meshgrid
    x = np.vstack([e.flat for e in mg]).T
    return np.array([func(i) for i in x]).reshape(mg[0].shape)

def cont(f,  coord, grid_density=100):
    fig = plt.figure()
    
    #
    xx = np.linspace(coord[0][0], coord[0][1], grid_density)
    yy = np.linspace(coord[1][0], coord[1][1], grid_density)
    X, Y = np.meshgrid(xx,yy)
    Z = apply_to_mg(f, X, Y)
    #assert()
    plt.contour(X,Y,exp(Z))
    #plt.xlim(xmin=coord[0][0], xmax=coord[0][1])
    #plt.ylim(ymin=coord[1][0], ymax=coord[1][1])
   # fig.show()

def visualize(f, s, coord ):
    (xin, yin) = s.T
    fig = plt.figure()
    
    xx = np.linspace(coord[0][0], coord[0][1],100)
    yy = np.linspace(coord[1][0], coord[1][1],100)
    X, Y = np.meshgrid(xx,yy)
    Z = apply_to_mg(f, X, Y)
    plt.contour(X,Y,exp(Z),zorder=1)
    plt.scatter(xin, yin,zorder=2)
    
    #assert()
    
    #plt.xlim(xmin=coord[0][0], xmax=coord[0][1])
    #plt.ylim(ymin=coord[1][0], ymax=coord[1][1])
    fig.show()


def one_over_4th_root_t_schedule(t):
    return 1. / np.sqrt(2 + t)



def get_StaticLangevin(D, target_log_pdf, grad):
    
    step_size = 0.5
    acc_star = 0.56
    gamma2 = 0.1
    instance = StaticLangevin(D, target_log_pdf, grad, step_size, one_over_4th_root_t_schedule, acc_star)
    instance.manual_gradient_step_size = 0.5

    return instance

def get_AdaptiveLangevin(D, target_log_pdf, grad, prec = False, step_size=1):
    

    acc_star = 0.574
    gamma2 = 0.1
    instance = AdaptiveLangevin(D, target_log_pdf, grad, step_size, one_over_4th_root_t_schedule, acc_star)
    instance.do_preconditioning = prec
    instance.manual_gradient_step_size = 1
    return instance


if __name__ == '__main__':
    Log.set_loglevel(20)
    max_moment = 8
    D = 10
    pop_size=10
    bananicity = 0.03
    V = 100
    Z = sample_banana(3000, D, bananicity, V)
    moments = np.array([(Z**i).mean(0) for i in range(1, max_moment)])
    pk = PolynomialKernel(3)
    print(moments)
    true_correct = np.linalg.inv(np.cov(Z.T))
    target_log_pdf = lambda x: log_banana_pdf(x, bananicity, V, compute_grad=False)
    target_grad = lambda x: log_banana_pdf(x, bananicity, V, compute_grad=True)

    samplers = [
#                
                
                ]
    
    sampler_is = get_StaticLangevin(D, target_log_pdf, target_grad)#get_AdaptiveLangevin(D, target_log_pdf, target_grad)
    sampler_mh = get_StaticLangevin(D, target_log_pdf, target_grad)#get_AdaptiveLangevin(D, target_log_pdf, target_grad, prec=True, step_size=1.)
    start = np.zeros(D)
    num_iter = 100
    
    samples, log_target_densities, unadj_samp, unadj_log_target, logw, unw_samptimes = mini_rb_pmc(sampler_is, start, num_iter, pop_size, D, time_budget=100000)
    mom_gen = np.mean((np.array([(samples**i).mean(0) for i in range(1, max_moment)]) - moments)**2)    
    mcmc_samps = mini_mcmc(sampler_mh, start, num_iter, D)
    
    #the weights we get back are not Rao-Blackwellized, which is what we do now.
    #beware: this only works if the proposal is not adapted during sampling!!
    #logw = logsumexp(np.array([sampler_is.proposal_log_pdf(i, unadj_samp) for i in unadj_samp]), 0)
    
    res_idx = system_res(range(len(logw)), logw, resampled_size=10*len(logw))
    samples = unadj_samp[res_idx]
    mom_unadj = np.mean((np.array([(unadj_samp**i).mean(0) for i in range(1, max_moment)]) - moments)**2)
    mom_w = np.mean((np.array([(unadj_samp**i * exp(logw - logsumexp(logw))[:,np.newaxis]).sum(0) for i in range(1, max_moment)]) -moments)**2)
    mom_mcmc = np.mean((np.array([(mcmc_samps[0]**i).mean(0) for i in range(1, max_moment)]) - moments)**2)
    
    if False:
        plt.scatter(samples.T[0], samples.T[1], c='r', marker='*', zorder=4, s=5)
    #    fig.suptitle("%s - importance resampled" %  (sampler_is.__class__.__name__,))
        plt.show()
        plt.scatter(unadj_samp.T[0], unadj_samp.T[1], c = logw - logsumexp(logw), cmap = plt.get_cmap('Blues'), alpha=0.5, zorder=2) #)visualize_scatter_2d()
    #    plt.suptitle("%s - unadjusted Langevin" %  (sampler_is.__class__.__name__,))
    #    plt.scatter(mcmc_samps[0].T[0], mcmc_samps[0].T[1], c='b',marker='*')
        plt.show()
    Log.get_logger().info('===='+str(sampler_mh.step_size)+' '+str(mcmc_samps[2].mean())+'====')
    #the following two should be 0 ideally
    #mmd_w = pk.estimateMMD(Z,samples)
    #mmd_unw = pk.estimateMMD(Z,unadj_samp)
    Log.get_logger().info('estimated weighted mse moments %.2f, res per gen %.2f unadj %.2f, mcmc %.2f\n weight/unw. (good if <1):%.2f\n is/mcmc %2f' % (mom_w, mom_gen, mom_unadj,mom_mcmc, mom_w/mom_unadj, mom_w/mom_mcmc))

#    plt.show()

