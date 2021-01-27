import math
import torch
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, LambdaLR

'''
torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1, verbose=False):
    # sets the learning rate of each parameter group to the initial lr times a given function
    Args:
    lr_lambda : a function which computes a multiplicative factor
    last_epoch: the index of last epoch
    verbose: if True, prints a message to stdout for each update (default: Fasle)
    def get_lr(self):
        return [base_lr *lmbda(self.last_epoch) ...]

torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=01, verbose=False):
    Args:
    optimizer: period of learning rate decay
    gamma: multiplicative factor of learning rate decay (default: 0.1)

    def get_lr(self):
        return group['lr'] * self.gamma

'''
class StepLR(StepLR):
    @classmethod
    def resolve_args(cls, args, optimizer = optim.Optimizer):
        options = {}
        options['step_size'] = args.get("step_size", 0) 
        options['gamma'] = args.get("gamma", 0.1)
        return cls(optimizer, **options)

'''linear warm up for the first 10 epochs, cosine decay schedule, no restarts'''
class SimclrLR(LambdaLR):
    def __init__(self, optimizer, warm_up, epochs, cycles):
        self.warm_up = warm_up
        self.epochs = epochs
        self.cycles = cycles
        #self.min_lr = min_lr

        super(SimclrLR, self).__init__(optimizer, self.lr_lambda)
    
    @classmethod
    def resolve_args(cls, args, optimizer = optim.Optimizer):
        options = {}
        options['warm_up'] = args.get("warm_up", 10)
        options['epochs'] = args.get("max_epochs", 100)
        options['cycles'] = args.get("cycles", 1)
        #options['min_lr'] = args.get("min_lr", 1e-4)
        #options['last_epoch'] = args.get("last_eoch", -1)
        return cls(optimizer, **options)

    def lr_lambda(self, step):
        if step.self.warm_up:
            return float(step) / float(max(1.0, self.warm_up))
        # progress after warmup
        progress = float(step - self.warm_up) / float(max(1.0, self.epochs-self.warm_up))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
