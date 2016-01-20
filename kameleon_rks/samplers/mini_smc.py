from __future__ import print_function, absolute_import, division

from numpy import exp, log
from scipy.misc import logsumexp

from kameleon_rks.samplers.tools import system_res
from kameleon_rks.tools.log import Log
import numpy as np

logger = Log.get_logger()

def mini_smc(num_samples,  # will give size of sample in final iteration
                      population_size,
                      prior,  # some distribution object that our algorithm will have no problem with
                      log_targ,  # actual target
                      proposal_obj,
                      targ_ef_bridge=0.5,
                      targ_ef_stop=0.9,
                      ef_tolerance=0.02,
                      reweight=False,
                      across=False,
                      estim_evid=False,
                      ess=False):
    # ToDO: - adaptive resampling (only use importance resampling if ESS < threshold)
    #      - reweight earlier iterations for actual target
    #      - use weighted approximation instead of approx after resampling for final iteration
    """
    Sample from a geometric sequence of target distributions between prior and target,
    reweighting samples from early iterations for estimating the actual target.
    Uses a geometric bridge for now
    
    
    = Parameters =
    
    num_samples     - size of sample in final iteration
    population_size - size of particle system except for final iteration
    prior           - some easy target distribution, prior will likely do well
    log_target      - actual target
    proposal_obj    - object with which proposals are generated
    targ_ef_bridge  - target efficiency factor for bridge, i.e. what should eff/num_particles be in a bridge step
    targ_ef_stop    - target efficiency factor with respect to final target
    ef_tolerance    - efficiency factor tolerance    
    reweight        - False (only use last iteration), True (reweight for actual target and weight iterations by ESS)
    across          - Resampling across iterations after reweighting? True (resample. only if reweight = True)
    estim_evid      - return estimate of evidence/normalizing constant of log_target
    ess             - Return ESS of last iteration? Defaults to False.
    """
    
    logger.info("Starting SMC using %s" % \
            (proposal_obj.__class__.__name__))
    
    if not reweight:
        assert(not across)
        
        
    log_target = lambda x: np.apply_along_axis(lambda y: np.atleast_1d(log_targ(y)), 1, x)

    

    
    initial_guesses = prior.rvs(population_size)
    
    population_size = initial_guesses.shape[0]
    dim = initial_guesses.shape[1]
    
    lprior = np.empty(num_samples + population_size * 3)
    lprior[:population_size] = prior.logpdf(initial_guesses)
    
    lpost = np.empty(num_samples + population_size * 3)
    lpost[:population_size] = log_target(initial_guesses).flatten()
    
    rval = np.r_[initial_guesses, np.zeros((num_samples + population_size * 3, dim))]
    
    def ensure(size):
        old = len(lprior)
        if old < size:
            lprior.resize(old * 2)
            lpost.resize(old * 2)
            rval.resize((old * 2, rval.shape[1]))
    
    def seq_value(br, prior_value, posterior_value):
        """
        Compute logprobability from lprior and lpost according to
        bridge parameter/temperature br
        
        
        = Parameters =
        
        br              - bridge parameter/temperature
        prior_value     - value according to br = 0
        posterior_value - value according to br = 1
        """
        return prior_value * (1. - br) + posterior_value * br
        
    
    def incr_weight(idx_from, idx_to, br_old, br_new, return_ef=False):
        inc_w = (seq_value(br_new, lprior[idx_from:idx_to], lpost[idx_from:idx_to])  # use lpost[idx_beg:idx_mid] here for approximating actual target
                - seq_value(br_old, lprior[idx_from:idx_to], lpost[idx_from:idx_to]))
        assert(not np.any(np.isnan(rval)))
        if return_ef:
            norm_inc_w = inc_w - logsumexp(inc_w)
            ESS = exp(2 * logsumexp(norm_inc_w) - logsumexp(2 * norm_inc_w))
            EF = ESS / (idx_to - idx_from)
            return (inc_w, EF)
        return inc_w
    
        
                 
    def mcmc_rejuvenate(cur_sample, cur_lprior, cur_lpost, br):
        """
        Make an MCMC move using proposal_obj, overwriting input (except br)
        Returns the acceptance probabilities.
        
        
        = Parameters =
        
        cur_sample  - the particles to be moved (will be overwritten with new state after move)
        cur_lpost   - the logposteriors according to final target distribution in distribution sequence (will be overwritten)
        cur_lprior  - the logpriors according to first target distribution in distribution sequence (will be overwritten)
        br          - the bridge parameter/temperature which determines how posterior and prior are mixed for current target distribution in the sequence
        
        = Return =
        
        Acceptance probabilites
        """
        
        # set the target to be the intermediary distribution
        save_target_logpdf = proposal_obj.target_log_pdf
        proposal_obj.target_log_pdf = lambda x:seq_value(br, prior.logpdf(x), save_target_logpdf(x))
        if proposal_obj.__dict__.has_key('target_grad'):
            save_target_grad = proposal_obj.target_grad
            proposal_obj.target_grad = lambda x:seq_value(br, prior.logpdf_grad(x), save_target_grad(x))
            
        tmp = [proposal_obj.proposal(cur_sample[idx], seq_value(br, cur_lprior[idx], cur_lpost[idx])) for idx in range(len(cur_sample))]
        
        
        # reset the target to be the actual posterior
        proposal_obj.target_log_pdf = save_target_logpdf
        if proposal_obj.__dict__.has_key('target_grad'):
            proposal_obj.target_grad = save_target_grad
            
        # (prop, lprob_move_forw, lprob_move_back) = [np.array(l) for l in
        #                                    zip(*tmp)]
        (prop, lprob_bridge_forw, _, lprob_move_forw, lprob_move_back, current_kwargs) = [np.array(l) for l in
                                            zip(*tmp)]
                                                      
        
        # compute log_target for proposals
        lprior_forw = prior.logpdf(prop).flatten()
        lpost_forw = (lprob_bridge_forw.flatten() - (1 - br) * lprior_forw.flatten()) / br
        assert(np.allclose(lpost_forw, log_target(prop).flatten()))
      
        # compute all acceptance probabilites
        assert(not (np.any(np.isinf(lprob_move_forw)) or np.any(np.isnan(lprob_move_forw))))
        assert(not (np.any(np.isnan(lprior_forw))))
        assert(not (np.any(np.isnan(lpost_forw))))
        assert(not (np.any(np.isinf(lprob_move_back)) or np.any(np.isnan(lprob_move_back))))
        assert(not (np.any(np.isinf(cur_lprior)) or np.any(np.isnan(cur_lprior))))
        assert(not (np.any(np.isinf(cur_lpost)) or np.any(np.isnan(cur_lpost))))
        mh_ratio = (lprob_move_back + seq_value(br, lprior_forw, lpost_forw)
                         - lprob_move_forw - seq_value(br, cur_lprior.flatten(), cur_lpost.flatten()))
        assert(mh_ratio.shape == lpost_forw.shape)
        acc = exp(np.min(np.c_[np.zeros_like(mh_ratio), mh_ratio], 1))
        assert(not(np.any(np.isnan(acc)) or np.any(np.isinf(acc))))
        move = np.random.rand(len(acc)) < acc
        assert(np.mean(acc) != 0)
        
        cur_sample[:] = prop * np.atleast_2d(move).T + cur_sample * (1 - np.atleast_2d(move).T)
        cur_lpost[:] = lpost_forw * move + cur_lpost * (1 - move)
        cur_lprior[:] = prior.logpdf(cur_sample)  # lprior_forw*move + cur_lprior*(1-move)
        
        return acc
    
    def search_bridge_param(target_ef_fact, idx_beg, idx_end, br_old, eps=ef_tolerance):
        high = 1.0
        low = br_old
        
        max_eval = 9
        old_EF = 0
        
        logger.debug('Start bridge search')
        for i in range(max_eval + 1):
            mid = low + (high - low) / 2
            (inc_w, EF) = incr_weight(idx_beg, idx_end, br_old, mid, True)
            logger.debug(EF)
            d = EF - target_ef_fact
            if i == max_eval or np.abs(EF - old_EF) < eps:
                return (mid, inc_w)
            old_EF = EF
            if d < -eps:
                high = mid
            elif d > eps:
                low = mid
            else:
                return (mid, inc_w)
                
    
    def smc_iteration(beg, mid, end, br_old, target_ef_fact):
        (br_new, inc_w) = search_bridge_param(target_ef_fact, beg, mid, br_old)
        samps_idx = (np.array(system_res(range(population_size), resampled_size=end - mid, weights=inc_w))
                     + beg)

        rval[mid:end] = rval[samps_idx]
        lpost[mid:end] = lpost[samps_idx]
        lprior[mid:end] = lprior[samps_idx]
        
        proposal_obj.set_batch(rval[samps_idx])
        # pre = (rval[mid:end].copy(),  lprior[mid:end].copy(), lpost[mid:end].copy())
        acc = mcmc_rejuvenate(rval[mid:end], lprior[mid:end], lpost[mid:end], br_new)
        mean_acc = np.mean(acc)
        
        proposal_obj.next_iteration()
        proposal_obj.update_step_size([mean_acc])
        return (br_new, inc_w, mean_acc)
        
    br = [0.0]
    evid = 0
    
    old_EF = 0
    j = 1
    
    while True:
        idx_beg = (j - 1) * population_size
        idx_mid = idx_beg + population_size
        idx_end = idx_mid + population_size
        ensure(idx_end)
        
        (br_new, inc_w, mean_acc) = smc_iteration(idx_beg, idx_mid, idx_end, br[j - 1], targ_ef_bridge)
        br.append(br_new)
        
        evid = evid + logsumexp(inc_w) - log(inc_w.size)
        norm_inc_w = inc_w - logsumexp(inc_w)
        j = j + 1
        
        ESS = exp(2 * logsumexp(norm_inc_w) - logsumexp(2 * norm_inc_w))
        logger.debug("At bridge distribution #%d, ESS: %.2f, mean acc: %.4f, step_size: %.4e" % 
              (j, ESS, mean_acc, proposal_obj.step_size))
        
        # test how good we are  with respect to the actual distribution of interest
        (inc_w_final, EF_final) = incr_weight(idx_mid, idx_end, br[j - 1], 1, True)
        if (np.abs(EF_final - old_EF) < ef_tolerance or  # we're not improving much
            np.abs(EF_final - targ_ef_stop) < ef_tolerance):  # we reached our desired efficiency factor
            break
        old_EF = EF_final


    idx_beg = (len(br) - 1) * population_size
    idx_mid = idx_beg + population_size
    idx_end = idx_mid + num_samples
    ensure(idx_end)
    (br_new, inc_w, mean_acc) = smc_iteration(idx_beg, idx_mid, idx_end, br[-1], 1)
    br.append(br_new)
    evid = evid + logsumexp(inc_w) - log(inc_w.size)
    norm_inc_w = inc_w - logsumexp(inc_w)
    ESS = exp(2 * logsumexp(norm_inc_w) - logsumexp(2 * norm_inc_w))
    
    logger.debug("Final approx, ESS: %.2f, mean acc: %.4f" % 
          (ESS, mean_acc))

    if reweight == False:
        (rval, lpost) = (rval[idx_mid:idx_end], lpost[idx_mid:idx_end])
    else:
        # reweight for actual target
        power = 1. - np.repeat(br, population_size)
        rew = (lpost[:idx_mid] - lprior[:idx_mid]) * power
        
        # weight each iteration by ESS wrt actual target
        rew_resh = rew.reshape((len(br), population_size))
        logess = ((2 * logsumexp(rew_resh, 1) - logsumexp(2 * rew_resh, 1)))
        print(exp(logess))
        logess_sampsize_ratio = logess - log(population_size)
        rew = rew + np.repeat(logess_sampsize_ratio.flatten(), population_size)
        
        smp_idx = np.array(system_res(range(idx_mid), resampled_size=population_size, weights=rew))
        
        if across:
            smp_idx_acr = np.array(system_res(range(idx_end), resampled_size=idx_end, weights=np.r_[rew, np.ones(idx_end - idx_mid)]))
            
            (rval_acr, lpost_acr) = (rval[smp_idx_acr], lpost[smp_idx_acr])
        (rval, lpost) = (np.r_[rval[smp_idx], rval[idx_mid:idx_end]], np.r_[lpost[smp_idx], lpost[idx_mid:idx_end]])

    all_rval = [rval, lpost]  
    if ess:
        all_rval.append(ESS)
    if across:
        all_rval.extend((rval_acr, lpost_acr))
    if evid:
        all_rval.append(evid)
        
    return all_rval
