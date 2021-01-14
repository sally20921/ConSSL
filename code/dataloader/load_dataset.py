from collections import defaultdict

import torch

from utils import *
from .preprocess_image import preprocess_images

import os
import re
from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset, DataLoader

modes = ['train', 'traintest','val', 'test']

def load_data(args):
    print('Loading image data')

    train_iter, traintest_iter, val_iter = preprocess_images(args)

    return {'train': train_iter, 'traintest': traintest_iter, 'val': val_iter, 'test': val_iter}

def get_iterator(args):
    iters = load_data(args)
    print("Data loading done")

    return iters



