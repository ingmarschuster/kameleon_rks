from __future__ import print_function, division, absolute_import

import time

from kameleon_rks.samplers.tools import system_res, compute_ess
from kameleon_rks.tools.log import Log
import numpy as np

from numpy import exp, log, sqrt
from scipy.misc import logsumexp

logger = Log.get_logger()


class JointProposal(object):
    def __init__(self, transition_kernel, centers, logweights = None):
        if len(centers.shape) < 2:
            self.centers = centers[np.newaxis, :]
        else:
            self.centers = centers
            
        if logweights is None:
            self.lw = np.ones(len(self.centers)) - log(len(self.centers))
        else:
            self.lw = logweights - logsumexp(logweights)
        self.kern = transition_kernel
        
        
    def logp(self, proposals):
        lp_per_center = np.array([self.lw[i]+self.kern.proposal_log_pdf(cent, proposals) for i, cent in self.centers.enumerate()])
        return logsumexp(lp_per_center, 0)
    
    def rvs(self, num_samps):
        res_idx = system_res(range(len(self.lw)), self.lw)
        rval = np.zeros((num_samps, self.centers.shape[1]))
        for i in range(num_samps):
            rval[i], _, _, _, _, _ = self.kern.proposal(self.centers[res_idx[i]], -4, **{})
        return rval
            


def mini_rb_pmc(transition_kernel, start, num_iter, pop_size, D, recompute_log_pdf=False, time_budget=None, ):
    # PMC results
    print(pop_size)
    assert(num_iter % pop_size == 0)
    
    # following not implemented yet
    assert(recompute_log_pdf == False)
    
    proposals = np.zeros((num_iter // pop_size, pop_size, D)) + np.nan
    logweights = np.zeros((num_iter // pop_size, pop_size)) + np.nan
    prop_target_logpdf = np.zeros((num_iter // pop_size, pop_size)) + np.nan
    prop_prob_logpdf = np.zeros((num_iter // pop_size, pop_size)) + np.nan
    
    samples = np.zeros((num_iter, D)) + np.nan
    log_pdf = np.zeros(num_iter) + np.nan
    
    # timings for output and time limit
    times = np.zeros(num_iter)
    
    # for adaptive transition kernels
    avg_accept = 0.
    
    current = np.array([start]*pop_size)
    current_log_pdf = np.zeros(pop_size)+np.nan
    acc_prob = np.zeros(pop_size) + np.nan
    
    logger.info("Starting PMC using %s in D=%d dimensions" % \
                (transition_kernel.__class__.__name__, D,))
    it = 0
    
    for stage in range(num_iter // pop_size):
        start_it = stage * pop_size
        # stop sampling if time budget exceeded
        if time_budget is not None and not np.isnan(times[start_it]):
            if times[start_it] > times[0] + time_budget:
                logger.info("Time limit of %ds exceeded. Stopping MCMC at iteration %d." % (time_budget, it))
                break
            # print  progress
            if False and stage > 1:
                log_str = "PMC iteration %d/%d, current log_pdf: %.6f, avg acceptance: %.3f" % (it + 1, num_iter,
                                                                           np.nan if log_pdf[it - 1] is None else log_pdf[it - 1],
                                                                           avg_accept)
                logger.info(log_str)
        range_it = range(start_it, start_it + pop_size)
        for it in range_it:
            prop_idx = it - start_it
            if np.isnan(current_log_pdf[prop_idx]):
                cur_lpdf = None
            else:
                cur_lpdf = current_log_pdf[prop_idx]
            times[it] = time.time()            
            # marginal sampler: make transition kernel re-compute log_pdf of current state
            if recompute_log_pdf:
                current_log_pdf = None
            
            # generate proposal and acceptance probability
            logger.debug("Performing GRIS sample %d" % it)
            proposals[stage, prop_idx], prop_target_logpdf[stage, prop_idx], current_log_pdf[prop_idx], prop_prob_logpdf[stage, prop_idx], backw_logpdf, current_kwargs = transition_kernel.proposal(current[prop_idx], cur_lpdf, **{})
            #logweights[stage, prop_idx] = prop_target_logpdf[stage, prop_idx] - prop_prob_logpdf[stage, prop_idx]

        #Rao-Blackwellize over all used proposals
        all_prop_logpdfs = np.array([transition_kernel.proposal_log_pdf(current[it - start_it], proposals[stage, :]) for it in range_it])
        prop_prob_logpdf[stage, :] = logsumexp(all_prop_logpdfs, 0)
        logweights[stage, :] = prop_target_logpdf[stage, :] - prop_prob_logpdf[stage, :]
        res_idx = system_res(range(pop_size), logweights[stage, :])
        samples[range_it], log_pdf[range_it] = proposals[stage, res_idx], prop_prob_logpdf[stage, res_idx]
        ess = compute_ess(logweights[stage, :], normalize=True)
        if ess/float(pop_size) > 0.5:
            current = proposals[stage, :]
            current_log_pdf = prop_target_logpdf[stage, :]
        else:
            current = proposals[stage, res_idx]
            current_log_pdf = prop_prob_logpdf[stage, res_idx]
        #print(ess, float(pop_size)/ess)
        transition_kernel.next_iteration()
        transition_kernel.update(np.vstack(proposals[:stage+1, :]), pop_size, np.hstack(logweights[:stage+1, :])) 

        
        
    res_idx = system_res(range(pop_size*stage)*10, weights=logweights[:stage, :].flatten()) 
    unw_samp = np.vstack(proposals[:pop_size*stage])
    unw_logtarg = np.hstack(prop_target_logpdf[:pop_size*stage])
    
    # recall it might be less than last iterations due to time budget
    return  samples[:it], log_pdf[:it], unw_samp, unw_logtarg, np.hstack(logweights[:pop_size*stage]), times[:it]