from pathlib import Path
import random

from fire import Fire
from munch import Munch

import torch
import numpy as np

from config import config, debug_options
from dataloader.load_dataset import get_iterator
from utils import wait_for_key, suppress_stdout
from train import train
#from evaluate import evaluate
#from infer import infer 

class Cli:
    def __init__(self):
        self.defaults = config
        self.debug = debug_options

    # update arguments if any argument was given from command
    def _default_args(self, **kwargs):
        args = self.defaults
        if 'debug' in kwargs:
            args.update(self.debug)
        args.update(kwargs)
        
        #update arguments to its absolute path
        args.update(resolve_paths(config))
        args.update(fix_seed(args))
        args.update(get_device(args))
        
        # at the end, print all the updated arguments
        print(args)

        return Munch(args)

    # the most important part, checking dataloader dir
    def check_dataloader(self, **kwargs):
        from dataloader.load_dataset import modes
        from utils import prepare_batch
        from tqdm import tqdm 
        print("check_dataloader")

        args = self._default_args(**kwargs)
        iters = get_iterator(args)
        #print(iters['train'])

        #for batch_idx, (inputs, targets) in enumerate(iters['train']):
        #    print("{}:({},{})".format(batch_idx, inputs.shape, targets.shape))
        #    106:(torch.Size([16, 3, 224, 224]),torch.Size([16]))

        #for batch in iters['train']:
        #    print('Test loading train data')
        #    batch = prepare_batch(args, batch)

        train_iter_test = next(iter(iters['train'])) # take out 1 batch
        # see if iteration in train proceeds well
        for (key1, key2), value in train_iter_test.items():
            if isinstance(value, torch.Tensor):
                print(key1, key2, value.shape)
            else:
                print(key1, key2, value)
        
        #print("test loading val data")
        #for batch_idx, batch in tqdm(iters['val']):
        #    batch = prepare_batch(args, batch)

        #for mode in modes:
        #    print('Test loading %s data' % mode)
        #    for batch_idx, batch in tqdm(iters[mode]):
        #    #    import ipdb; ipdb.set_trace() # XXX DEBUG
        #        batch = prepare_batch(args, batch)

    def train(self, **kwargs):
        args = self._default_args(**kwargs)
        train(args)
        wait_for_key()


def resolve_paths(config):
    paths = [k for k in config.keys() if k.endswith('_path')]
    res  = {}
    root = Path('../').resolve()
    for path in paths:
        res[path] = root / config[path]

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return {'device': device}

if __name__ == "__main__":
    Fire(Cli)
