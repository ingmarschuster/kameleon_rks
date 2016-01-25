from abc import abstractmethod

from kameleon_rks.tools.log import Log
import numpy as np


logger = Log.get_logger()

class ProposalBase():
    def __init__(self, D, target_log_pdf, step_size, schedule=None, acc_star=None, fixed_step_size=False):
        self.target_log_pdf = target_log_pdf
        self.D = D
        self.step_size = step_size
        self.schedule = schedule
        self.acc_star = acc_star
        self.fixed_step_size = fixed_step_size
        
        # some sanity checks
        assert acc_star is None or acc_star > 0 and acc_star < 1
        if schedule is not None:
            lmbdas = np.array([schedule(t) for t in  np.arange(100)])
            assert np.all(lmbdas > 0)
            assert np.allclose(np.sort(lmbdas)[::-1], lmbdas)
        
        self.t = 0
    
    def set_batch(self, Z):
        pass
    
    @abstractmethod
    def get_name(self):
        return self.__class__.__name__
    
    def mh(self, backward_log_pdf, forward_log_pdf, backward_log_prob, forward_log_prob):
        log_acc_prob = forward_log_pdf - backward_log_pdf + backward_log_prob - forward_log_prob
        return np.exp(np.min([0, log_acc_prob]))
    
    def update_step_size(self, previous_accept_probs):
        if self.schedule is not None and self.acc_star is not None and not self.fixed_step_size:
            old_step_size = self.step_size
            # generate updating weight
            lmbda = self.schedule(self.t)
            
            # difference desired and actual acceptance rate
            last_accept_prob = previous_accept_probs[-1]
            diff = last_accept_prob - self.acc_star
                
            self.step_size = np.exp(np.log(self.step_size) + lmbda * diff)
            
            logger.debug("Updated step_size from %.3f to %.3f using lmbda=%.3f" % (old_step_size, self.step_size, lmbda))

    def next_iteration(self):
        self.t += 1
        
    def update(self, Z, num_new = 1):
        pass
    
    def proposal(self, current, current_log_pdf,  **kwargs):
        pass
