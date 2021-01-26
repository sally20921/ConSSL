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
from ckpt import get_model_ckpt, save_ckpt
from model import get_model, get_linear
from loss import get_loss
from optimizer import get_optimizer, get_sub_optimizer, get_scheduler
from logger import get_logger, log_results, log_reults_cmd

from utils import _prepare_batch
from metric import get_metrics


def eval_linear(pretrain_args, args):
    # get pretrained model
    pt_args, pt_model, ckpt_available = get_model_ckpt(pretrain_args)
    
    tf = get_transform(args, 'train')
    ds = get_dataset(args, tf, 'train')

    if ckpt_available:
        print("loaded pretrained model {} in eval linear".format(args.ckpt_name))

    model = get_linear(args, pt_model, args.num_classes)
    loss_fn = get_loss(args)
    optimizer = get_sub_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    trainer = get_trainer(args, model, loss_fn, optimizer, scheduler)
    evaluator = get_evaluator(args, model, loss_fn)

    metrics = get_metrics(args)
    logger = get_logger(args)
    trainer.run(ds, max_epochs=args.epoch)

