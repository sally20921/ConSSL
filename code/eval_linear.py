'''
freeze the encoder and train the supervised classification head with a cross entropy loss
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np 

from ignite.engine.engine import Engine, State, Events
from ignite.metrics import Loss, Accuracy, TokKCategoricalAccuracy

from dataloader import get_transform
from dataloader import get_dataset

def eval_linear(pretrain_args, args):
    # get pretrained model
    pt_args, pt_model, ckpt_available = get_model_ckpt(pretrain_args)
    
    tf = get_transform(args, 'train')
    ds = get_dataset(args, tf, 'train')

    if ckpt_available:
        print("loaded pretrained model {}".format(args.ckpt_name))

    trainer.run(ds, max_epochs=args.epoch)

