import os
from pathlib import Path

from torch import optim

from inflection import underscore

optim_dict = {}

def add_optims():
    path = Path(os.path.dirname(__file__))

    for p in path.glob('*.py'):
        name = p.stem
        parent = p.parent.stem
        if name != "__init__":
            __import__("{}.{}".format(parent, name))
            module = eval(name)
            for member in dir(module):
                member = getattr(module, member)
                if hasattr(member, "__bases__") and ((optim.Optimizer in member.__bases__ or optim.lr_scheduler._LRScheduler in member.__bases__) or (optim.Optimizer in member.__bases__[0].__bases__ or optim.lr_scheduler._LRScheduler in member.__bases__[0].__bases__)):
                    optim_dict[underscore(str(member.__name__))] = member

def get_sub_optimizer(args, model):
    sub_optim = optim_dict[args.sub_optimizer]
    sub_optim = sub_optim.resolve_args(args, model.parameters())
    sub_optim.zero_grad()
    return sub_optim

def get_optimizer(args, sub_optimizer):
    optim = optim_dict[args.optimizer]
    optim = optim.resolve_args(args, sub_optimizer)
    optim.zero_grad()
    return optim

def get_scheduler(args, optimizer):
    # string value comparison
    #if args.stage == 'pretrain':
    scdl = optim_dict[args.scheduler]
    #else: # args.stage == linear_eval:
    #    scdl = optim_dict[args.linear_eval_scheduler]
    scdl = scdl.resolve_args(args, optimizer)
    #scdl.zero_grad()
    return scdl

add_optims()
