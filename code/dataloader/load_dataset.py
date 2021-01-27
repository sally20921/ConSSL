import math
import os

import PIL
from tqdm import tqdm

import torch
import torchvision
from torchvision import datasets

from utils import *

'''
dataset_types = [tiny_imagenet_dataset', 'imagenet_dataset', 'mnist_dataset', 'stl10_dataset', 'cifar10_dataset', 'cifar100_dataset']

I assumed we only use training data in the pretrain stage. If not, we need to reimplement it.
'''

class ImagenetDataset:
    def __init__(args, transform, eval_stage, bs, sf, nw):
        self.tf = transform
        self.st = eval_stage
        self.bs = bs
        self.sf = sf
        self.nw = nw

        if self.st == 'test':
            ip = os.path.join(args.image_path / 'train')
            self.loader = datasets.ImageFolder(ip, transform=self.tf)
        else:
            ip = os.path.join(args.image_path / 'val')
            self.loader = datasets.ImageFolder(ip, transform=self.tf)

    def __call__(self, x):
        return torch.utils.data.DataLoader(self.loader, batch_size=self.bs, shuffle=self.sf, num_workers=self.nw, pin_memory=True)

    @classmethod
    def resolve_args(self, args, transform, eval_stage):
        return cls(args, transform, eval_stage, args.batch, args.suffle, args.worker)

class TinyImagenetDataset:
    def __init__(args, transform, eval_stage, bs, sf, nw):
        self.tf = transform
        self.st = eval_stage
        self.bs = bs
        self.sf = sf
        self.nw = nw

        if self.st == 'test':
            ip = os.path.join(args.image_path / 'train')
            self.loader = datasets.ImageFolder(ip, transform=self.tf)
        else:
            ip = os.path.join(args.image_path / 'val')
            self.loader = datasets.ImageFolder(ip, transform=self.tf)

    def __call__(self, x):
        return torch.utils.data.DataLoader(self.loader, batch_size=self.bs, shuffle=self.sf, num_workers=self.nw, pin_memory=True)

    @classmethod
    def resolve_args(self, args, transform, eval_stage):
        return cls(args, transform, eval_stage, args.batch, args.suffle, args.worker)

class MNISTDataset:
    def __init__(args, transform, eval_stage, bs, sf, nw, ip):
        self.tf = transform
        self.st = eval_stage
        self.bs = bs
        self.sf = sf
        self.nw = nw
        self.ip = ip

        if self.st == 'test':
            self.loader = datasets.MNIST(self.ip, train=False, transform=self.tf, download=True)
            
        else:
            self.loader = datasets.MNIST(self.ip, train=True, transform=self.tf, download=True)

    def __call__(self, x):
        return torch.utils.data.DataLoader(self.loader, batch_size=self.bs, shuffle=self.sf, num_workers=self.nw, pin_memory=True)

    @classmethod
    def resolve_args(self, args, transform, eval_stage):
        return cls(args, transform, eval_stage, args.batch, args.suffle, args.worker, args.image_path)

class STL10Dataset:
    def __init__(args, transform, eval_stage, bs, sf, nw, ip):
        self.tf = transform
        self.st = eval_stage
        self.bs = bs
        self.sf = sf
        self.nw = nw
        self.ip = ip

        if self.st == 'test':
            self.loader = datasets.STL10(self.ip, train=False, transform=self.tf, download=True)
            
        else:
            self.loader = datasets.STL10(self.ip, train=True, transform=self.tf, download=True)

    def __call__(self, x):
        return torch.utils.data.DataLoader(self.loader, batch_size=self.bs, shuffle=self.sf, num_workers=self.nw, pin_memory=True)

    @classmethod
    def resolve_args(self, args, transform, eval_stage):
        return cls(args, transform, eval_stage, args.batch, args.suffle, args.worker, args.image_path)

class CIFAR10Dataset:
    def __init__(transform, eval_stage, bs, sf, nw, ip):
        self.tf = transform
        self.st = eval_stage
        self.bs = bs
        self.sf = sf
        self.nw = nw
        self.ip = ip

        if self.st == 'test':
            self.loader = datasets.CIFAR10(self.ip, train=False, transform=self.tf, download=True)
            
        else:
            self.loader = datasets.CIFAR10(self.ip, train=True, transform=self.tf, download=True)

    def __call__(self, x):
        return torch.utils.data.DataLoader(self.loader, batch_size=self.bs, shuffle=self.sf, num_workers=self.nw, pin_memory=True)

    @classmethod
    def resolve_args(cls, args, transform, eval_stage):
        return cls(transform, eval_stage, args.batch, args.shuffle, args.worker, args.image_path)

class CIFAR100Dataset:
    def __init__(args, transform, eval_stage, bs, sf, nw, ip):
        self.tf = transform
        self.st = eval_stage
        self.bs = bs
        self.sf = sf
        self.nw = nw
        self.ip = ip

        if self.st == 'test':
            self.loader = datasets.CIFAR100(self.ip, train=False, transform=self.tf, download=True)
            
        else:
            self.loader = datasets.CIFAR100(self.ip, train=True, transform=self.tf, download=True)

    def __call__(self, x):
        return torch.utils.data.DataLoader(self.loader, batch_size=self.bs, shuffle=self.sf, num_workers=self.nw, pin_memory=True)

    @classmethod
    def resolve_args(self, args, transform, eval_stage):
        return cls(args, transform, eval_stage, args.batch, args.suffle, args.worker, args.image_path)






        
