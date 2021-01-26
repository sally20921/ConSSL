from contextlib import contextmanager
from datetime import datetime

import os
import sys
import json
import pickle
import re

import six
import numpy as np
import torch


# for 1 batch
# output: {'x_i', 'x_j'}, target
# batch: (x_i, x_j), target
def prepare_batch(args, batch):
    net_input_key = [*args.use_inputs]
    net_input = {k: batch[0][i] for k, i in zip(net_input_key, range(len(net_input_key)))}
    for key, value in net_input.items():
        if torch.is_tensor(value):
            net_input[key] = value.to(args.device).contiguous()

    target = batch[1]
    if torch.is_tensor(target):
        target = target.to(args.device).contiguous()

    # return batch in output form 
    return net_input, target

# for 1 batch
# output: x, target
# batch: x, target
def _prepare_batch(args, batch):
    x, target = batch
    x = x.to(args.device).contiguous()
    target = target.to(args.device).contiguous()
    return x, target

