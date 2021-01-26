from contextlib import contextmanager
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.optim.optimizer import Optimizer


class LARS(Optimizer):
    def __init__(self, sub_optimizer, eps, trust_coefficient, clip):
        self.optim = sub_optimizer
        self.eps = eps
        self.trust_coef = trust_coefficient
        self.clip = clip

        @classmethod
    def resolve_args(cls, args, sub_optimizer):
        options = {}
        options['trust_coefficient'] = args.get("trust_coefficient", 0.001)
        options['eps'] = args.get("eps", 1e-8)
        options['clip'] = args.get("clip", True)
        return cls(sub_optimizer, **options)
        
    def __getstate__(self):
        larc_dict = {}
        larc_dict['eps'] = self.eps
        larc_dict['trust_coef'] = self.trust_coef
        larc_dict['clip'] = self.clip
        return (self.optim, larc_dict)

    def __setstate__(self, state):
        self.optim, larc_dict = state
        self.eps = larc_dict['eps']
        self.trust_coef = larc_dict['trust_coef']
        self.clip = larc_dict['clip']

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

    def step(self, *args, **kwargs):
        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)
                group['weight_decay'] = 0
                
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)

                    if param_norm != 0 and grad_norm != 0:
                        adaptive_lr = self.trust_coef * (param_norm) / (grad_norm + param_norm * weight_decay + self.eps)

                        if self.clip:
                            adaptive_lr = min(adaptive_lr/group[lr], 1) 
                            # calculation of adaptive_lr so that when multiplied by lr it equals `min(adaptive_lr, lr)`

                    p.grad.data += weight_decay * p.data
                    p.grad.data *= adaptive_lr

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[i]
        



    
    


