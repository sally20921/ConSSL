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

from config import log_keys

def load_json(path):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path, **kwargs):
    with open(path, 'w') as f:
        json.dump(data, f, **kwargs)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def get_dirname_from_args(args):
    dirname = ''
    for key in sorted(log_keys):
        dirname += '_'
        dirname += key
        dirname += '_'
        dirname += str(args[key])

    return dirname[1:]

def get_now():
    now = datetime.now()
    return now.strftime('%Y-%m-%d-%H-%M-%S')

'''torch.Tensor
contiguous(memory_format=torch.contiguous_format): returns a contiguous in memory tensor containing the same data as self tensor.
view() cannot be applied to discontiguous tensor.
'''
# for 1 batch 
# output: {'x_i': , 'x_j': }, target
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

def wait_for_key(key="y"):
    text = ""
    while (text != key):
        text = six.moves.input("Press {} to quit: ".format(key))
        if text == key:
            print("terminating process")
        else:
            print("key {} unrecognizable".format(key))

@contextmanager
def suppress_stdout(do=True):
    if do:
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
