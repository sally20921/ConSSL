'''
Munch is a Python dictionary that provides attribute-style access
'''

from pathlib import Path
import random
import numpy as np
from tqdm import tqdm

from fire import Fire
from munch import Munch

import torch

from dataloader import get_transform, get_dataset
from config import config, eval_linear, eval_semi, eval_transfer
from train import train
from eval_linear import eval_linear

from utils import wait_for_key, suppress_stdout, prepare_batch

class Cli:
    def __init__(self):
        self.cf = config 
        self.el = eval_linear
        self.es = eval_semi
        self.et = eval_transfer

    def pretrain(self, **kwargs):
        args = self.cf
        args.update(kwargs)
        args.update(resolve_path(args))
        args.update(fix_seed(args))
        args.update(get_device(args))

        print(args)

        args = Munch(args)
        pretrain(args)
        wait_for_key()

    def eval_linear(self, **kwargs):
        args = self.el
        args.update(kwargs)
        args.update(resolve_path(args))
        args.update(fix_seed(args))
        args.update(get_device(args))

        print(args)

        args = Munch(self.tf, args)
        eval_linear(args)
        wait_for_key()

def resolve_paths(args):
    paths = [k for k in args.keys() if k.endswith('_path')]
    res = {}
    root = Path('../').resolve()
    for path in paths:
        res[path] = root / args[path]

    return res

def fix_seed(args):
    if 'random_seed' not in args:
        args['random_seed'] = 0
    random.seed(args['random_seed'])
    np.random.seed(args['random_seed'])
    torch.manual_seed(args['random_seed'])
    torch.cuda.manual_seed_all(args['random_seed'])

    return args

def get_device(args):
    if hasattr(args, 'device'):
        device = args.device

    else:
        device = "cuda" if torch.cuda_is_available() else "cpu"

    return {'device': device}

def set_distributed(args):
    pass

if __name__=="__main__":
    Fire(Cli)


        




