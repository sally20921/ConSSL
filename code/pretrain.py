'''
get_transform(args, eval_stage)
get_dataset(args, transform, eval_stage)
In pretraining stage, eval_stage set to 'none'
'''
from dataloader import get_transform
from dataloader import get_dataset

from utils import prepare_batch
from logger import get_logger, log_results, log_results_cmd

from ignite.engine.engine import Engine, State, Events
from ignite.metric import Loss

import numpy as np
from apex import amp
import ignite.distributed as idist
from ignite.contrib.engines import common

def pretrain(args):
    tf = get_transform(args, 'none')
    ds = get_dataset(args, tf, 'none')


