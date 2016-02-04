from __future__ import print_function, division, absolute_import

import time

from kameleon_rks.samplers.tools import system_res
from kameleon_rks.tools.log import Log
from scipy.misc import logsumexp
import numpy as np


logger = Log.get_logger()

def mini_pmc(transition_kernel, start, num_iter, pop_size, recompute_log_pdf=False, time_budget=None, weighted_update=True, rao_blackwell_generation=True, resample_at_end=False):
    assert(len(start.shape) <= 2)

    
    
    assert(num_iter % pop_size == 0)
    
    # following not implemented yet
    assert(recompute_log_pdf == False)
    
    if len(start.shape) == 2:
        prev = start
        prev_logp = np.array([None] * start.shape[0])
        D = start.shape[1]
    else:
        prev = np.array([start] * pop_size)
        prev_logp = np.array([None] * pop_size)
        D = len(start)
    
    # PMC results
    proposals = np.zeros((num_iter // pop_size, pop_size, D)) + np.nan
    logweights = np.zeros((num_iter // pop_size, pop_size)) + np.nan
    prop_target_logpdf = np.zeros((num_iter // pop_size, pop_size)) + np.nan
    prop_prob_logpdf = np.zeros((num_iter // pop_size, pop_size)) + np.nan
    
    samples = np.zeros((num_iter, D)) + np.nan
    log_pdf = np.zeros(num_iter) + np.nan
    
    # timings for output and time limit
    times = np.zeros(num_iter)
    

    
    logger.info("Starting PMC using %s in D=%d dimensions using %d particles and %d iterations" % \
                (transition_kernel.get_name(), D, pop_size, num_iter / pop_size))
    it = 0
    
    time_last_printed = time.time()
    
    for stage in range(num_iter // pop_size):
        log_str = "PMC stage %d/%d" % (stage + 1, num_iter // pop_size)
        current_time = time.time()
        if current_time > time_last_printed + 5:
            logger.info(log_str)
            time_last_printed = current_time
        else:
            logger.debug(log_str)
        
        start_it = stage * pop_size
        # stop sampling if time budget exceeded
        if time_budget is not None and not np.isnan(times[start_it]):
            if times[start_it] > times[0] + time_budget:
                logger.info("Time limit of %ds exceeded. Stopping MCMC at iteration %d." % (time_budget, it))
                break
            # print  progress
#        if stage > 1:
#            log_str = "PMC iteration %d/%d, current log_pdf: %.6f" % (it + 1, num_iter,
#                                                                       np.nan if log_pdf[it - 1] is None else log_pdf[it - 1])
#            logger.debug(log_str)               

        range_it = range(start_it, start_it + pop_size)
        for it in range_it:
#            print(it)
            prop_idx = it - start_it
            times[it] = time.time()            
            # marginal sampler: make transition kernel re-compute log_pdf of current state
            if recompute_log_pdf:
                current_log_pdf = None
            
            # generate proposal and acceptance probability
            proposals[stage, prop_idx], prop_target_logpdf[stage, prop_idx], current_log_pdf, prop_prob_logpdf[stage, prop_idx], backw_logpdf, current_kwargs = transition_kernel.proposal(prev[prop_idx], prev_logp[prop_idx], **{})
            logweights[stage, prop_idx] = prop_target_logpdf[stage, prop_idx] - prop_prob_logpdf[stage, prop_idx]
        if rao_blackwell_generation:
            try:
                all_prop_logpdfs = np.array([transition_kernel.proposal_log_pdf(prev[it - start_it], proposals[stage, :]) for it in range_it])
                prop_prob_logpdf[stage, :] = logsumexp(all_prop_logpdfs, 0)
            except:
                assert()
        else:
            # print('norb')
            # assert()
            np.all(logweights[stage, :] == prop_target_logpdf[stage, :] - prop_prob_logpdf[stage, :])
        logweights[stage, :] = prop_target_logpdf[stage, :] - prop_prob_logpdf[stage, :]


        res_idx = system_res(range(pop_size), logweights[stage, :],)
        samples[range_it] = proposals[stage, res_idx]
        log_pdf[range_it] = prop_target_logpdf[stage, res_idx]
                
        prev = samples[range_it] 
        prev_logp = log_pdf[range_it]

        
        
        # update transition kernel, might do nothing
        transition_kernel.next_iteration()
        if weighted_update:
            transition_kernel.update(np.vstack(proposals[:stage + 1, :]), pop_size, np.hstack(logweights[:stage + 1, :])) 
        else:
            transition_kernel.update(samples[:range_it[-1] + 1], pop_size)
    
    if resample_at_end:
        #FIXME: the last stage might not have drawn the full set of samples
        all_lweights = np.hstack(logweights[:stage+1, :])
        res_idx = system_res(range(it+1), all_lweights)
        (samples, log_pdf) = (np.vstack(proposals[:stage+1, :])[res_idx], prop_target_logpdf[res_idx])
    else:
        samples, log_pdf = samples[:it], log_pdf[:it]

    # recall it might be less than last iterations due to time budget
    return samples, log_pdf, times[:it]
