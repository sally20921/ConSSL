import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True


