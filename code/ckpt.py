import os 

import torch, pickle
from torch import nn
import torch.nn.functional as F

from dataloader.load_dataset import get_iterator
from model import get_model
from utils import get_dirname_from_args

# how are we going to name our checkpoint file
def get_ckpt_path(args, epoch, loss):
    ckpt_name = get_dirname_from_args(args)
    # inside the ckpt path
    ckpt_path = args.ckpt_path / ckpt_name 
    # if you are creating checkpoint file for the first time
    args.ckpt_path.mkdir(exist_ok=True)
    ckpt_path.mkdir(exist_ok=True)

    # checkpoint name is named after the loss and epoch
    loss = '{:.4f}'.format(loss)
    ckpt_path = ckpt_path / 'loss_{}_epoch_{}.pickle'.format(loss, epoch)
    
    # return the path name/address
    return ckpt_path

# saving checkpoint file based on current status
def save_ckpt(args, epoch, loss, model):
    # since checkpoint file is named based on epoch and loss, we state which epoch is being saved 
    print('saving epoch {}'.format(epoch))

    dt = {
        'args': args,
        'epoch': epoch,
        'loss': loss,
        'model': model.state.dict(),
    }

    ckpt_path = get_ckpt_path(args, epoch, loss)
    # name checkpoint file based on epoch and loss
    print("Saving checkpoint {}".format(ckpt_path))
    # what checkpoint in what epoch

    torch.save(dt, str(ckpt_path))

# get a model from checkpoint file
def get_model_ckpt(args):
    # if there is a model specified to be fetched 
    ckpt_available = args.ckpt_name is not None

    if ckpt_available:
        name = '{}'.format(args.ckpt_name)
        # add * behind the name 
        name = '{}*'.format(name) if not name.endswith('*') else name
        # now every name has * behind it

        ckpt_paths = sorted(args.ckpt_path.glob(name), reverse=False)
        assert len(ckpt_paths>0), "no ckpt candidate for {}".format(args.ckpt_path / args.ckpt_name)
        # full address is ckpt_path / ckpt_name

        ckpt_path = ckpt_paths[0]
        print("loading from {}".format(ckpt_path))
        # load model from ckpt_path

        # 1. first update the arguments
        args.update(dt['args'])

    iters = get_iterator(args)
    # which iterator are we on
    # 2. get model based on the arguments
    model = get_model(args)

    if ckpt_available:
        model.load_state_dict(dt['model'])
        # load other state in the model 

    return args, model, iters, ckpt_available
        
