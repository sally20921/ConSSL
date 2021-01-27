''' 
SimCLR is optimized using LARS with linear learning rate scaling
(i.e. LearningRate = 0.3 * BatchSize / 256)
(i.e. weight decay = 1e-6)
(i.e. linear warmup for the first 10 epochs)
(i.e. lr_decay with the cosine decay schedule without restarts)
'''
import torch
from torch.optim.optimizer import Optimizer


class LARS(Optimizer):
    def __init__(self, sub_optimizer, eps, trust_coefficient):
        self.optim = sub_optimizer
        self.eps = eps
        self.trust_coef = trust_coefficient
        self.adaptive_lr = torch.ones([])

    @classmethod
    def resolve_args(cls, args, sub_optimizer):
        options = {}
        options['trust_coefficient'] = args.get("trust_coefficient", 0.001)
        options['eps'] = args.get("eps", 1e-8)
        return cls(sub_optimizer, **options)
        
    def __getstate__(self):
        lars_dict = {}
        lars_dict['eps'] = self.eps
        lars_dict['trust_coef'] = self.trust_coef
        lars_dict['adaptive_lr'] = self.adaptive_lr
        return (self.optim, lars_dict)

    def __setstate__(self, state):
        self.optim, lars_dict = state
        self.eps = lars_dict['eps']
        self.trust_coef = lars_dict['trust_coef']
        self.adaptive_lr = lars_dict['adaptive_lr']

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.optim)

    @property
    def param_groups(self):
        return self.optim.param_groups

    @property
    def state(self):
        return self.optim.state

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def apply_adaptive_lrs(self):
        with torch.no_grad():
            weight_decay = group['weight_decay']
            ignore = group.get('ignore', None) # this is set by add_weight_decay

        for p in group['params']:
            if p.grad is None:
                continue

        # add weight decay before computing adaptive LR
        # seems pretty important for SimCLR model
        if weight_decay > 0:
            p.grad = p.grad.add(p, alpha=weight_decay)

        if ignore is not None and not ignore:
            param_norm = p.norm()
            grad_norm = p.grad.norm()

            # compute adaptive learning rate
            adaptive_lr = 1.0
            if param_norm > 0 and grad_norm > 0:
                adaptive_lr = self.trust_coef * param_norm / (grad_norm + self.eps)
            
            print("applying {} lr scaling to param of shape {}".format(adaptive_lr, p.shape))
            p.grad = p.grad.mul(adaptive_lr)

    def step(self, *args, **kwargs):
        self.apply_adaptive_lrs()

        # zero out weight decay 
        weight_decay  = [group['weight_decay'] for group in self.optim.param_groups]
        for group in self.optim.param_groups:
            group['weight_decay'] = 0

        loss = self.optim.step(*args, **kwargs) # sub optimizer

        # restore weight decay
        for group, wo in zip(self.optim.param_groups, weight_decay):
            group['weight_decay'] = wo
        
        return loss 



    
    


