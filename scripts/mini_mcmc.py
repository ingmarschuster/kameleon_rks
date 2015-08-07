import time

import numpy as np


def mini_mcmc(sampler, start, num_iter, target_log_pdf, D, time_budget = None, quiet = False):
    # MCMC results
    samples = np.zeros((num_iter, D))
    proposals = np.zeros((num_iter, D))
    log_probs_proposals = np.zeros(num_iter)
    log_probs = np.zeros(num_iter)
    acceptance_log_probs = np.zeros(num_iter)
    accepteds = np.zeros(num_iter)
    times = np.zeros(num_iter)
    
    last_time_printed = time.time()
    
    # init MCMC (first iteration)
    current = start.copy()
    current_target_log_prob = target_log_pdf(current)
    for it in range(num_iter):
        times[it] = time.time()
        
        # stop sampling if time budget exceeded
        if time_budget is not None:
            if times[it] > times[0] + time_budget:
                print("Time limit of %ds exceeded. Stopping MCMC at iteration %d." % (time_budget, it))
                break
        
        # print chain progress
        if times[it] > last_time_printed + 5:
            last_time_printed = times[it]
            if not quiet:
                log_str = "%.1fs, MCMC iteration %d/%d, current log_pdf: %.6f, avg acceptance=%.3f" % (times[it]-times[0], it + 1, num_iter,
                                                                           log_probs[it - 1],
                                                                           np.mean(accepteds[:it]))
                print(log_str)
        
        
        # propose
        proposal, proposal_log_prob, proposal_log_prob_inv = sampler.proposal(current)
        proposal_target_log_prob = target_log_pdf(proposal)
        
        # Metropolis-Hastings acceptance probability
        accept_log_prob = proposal_target_log_prob - current_target_log_prob + \
                            proposal_log_prob_inv - proposal_log_prob
        accept_log_prob = np.min([0., accept_log_prob])
        
        # accept-reject
        accepted = np.log(np.random.rand()) <= accept_log_prob
        if accepted:
            current = proposal
            current_target_log_prob = proposal_target_log_prob
        
        # store results for this iteration
        samples[it] = current
        proposals[it] = proposal
        log_probs_proposals[it] = proposal_target_log_prob
        log_probs[it] = current_target_log_prob
        acceptance_log_probs[it] = accept_log_prob
        accepteds[it] = accepted
        
        # update adaptive sampler
        sampler.update(current, np.exp(accept_log_prob))
    
    # recall it might be less than last iterations due to time budget
    return samples[:it], proposals[:it], log_probs_proposals[:it], log_probs[:it], acceptance_log_probs[:it], accepteds[:it], times[:it]