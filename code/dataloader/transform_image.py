'''
transform_types = ['eval_transform', 'moco_transform', 'simclr_transform', 'byol_transfrom', 'swav_transform', 'sim_siam_transform']

In pretraining, pick between 'moco_transfrom' ~ 'sim_siam_transform'. These return two crops of the same image.
In eval_linear, eval_semi, eval_transfer, pick 'eval_transform'. This returns one crop of the image.

imagenet_norm = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
cifar_norm = [[0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]]
'''
# Must include 'Transform' at the end, must be a class, include resolve_args
import math
import os

import PIL 
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T
from torchvision.transforms import GaussianBlur

from utils import *

class EvalTransform:
    def __init__(args, eval_stage): # this stage is different from the whole stage, this just tells train/test
        self.im = args.image_size
        self.norm = self.normalize
        self.stage = eval_stage

        if self.stage == 'train': 
            self.transform = T.Compose([
                T.RandomResizedCrop(self.im, scale=(0.08), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(*self.norm)
            ])
        elif self.stage == 'test':
            self.transform = T.Compose([
                T.Resize(int(self.im*(8/7)), interpolation=Image.BICUBIC),
                T.CenterCrop(self.im),
                T.ToTensor(),
                T.Normalize(*self.norm)
            ])
        
        else:
            print("EvalTransform eval_stage train nor test. check spelling.")
    
    def __call__(self, x):
        return self.transform(x)

    @classmethod
    def resolve_args(cls, args, eval_stage):
        return cls(args=args, eval_stage=eval_stage)

class SimclrTransform:
    def __init__(self, args):
        self.im = args.image_size
        self.norm = args.normalize
        self.transform = T.Compose([
            T.RandomResizedCrop(self.im, scale=(0.2,1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.8,0.8,0.8,0.2)],p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=self.im//20*2+1, sigma=(0.1,2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize(*self.norm)
        ])

    def __call__(self, x):
        return self.transform(x), self.tranform(x)

    @classmethod
    def resolve_args(cls, args, eval_stage):
        return cls(args=args)

class BYOLTransform:
    def __init__(self, args):
        self.im = args.image_size
        self.norm = args.normalize
        self.transform = T.Compose([
            T.RandomResizedCrop(self.im, scale=(0.2,1.0)),
            T.RandomHorizontalFlip(),
            T.RnadomApply([T.ColorJitter(0.8,0.8,0.8,0.2)],p=0.8),
            T.RandomGrayScale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=self.im//20*2+1, sigma=(0.1,2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize(*self.norm)
        ])

    def __call__(self, x):
        return self.transform(x), self.tranform(x)

    @classmethod
    def resolve_args(cls, args, eval_stage):
        return cls(args=args)


class SimSiamTransform:
    def __init__(self, args):
        self.im = args.image_size
        self.norm = args.normalize
        self.transform = T.Compose([
            T.RandomResizedCrop(self.im, scale=(0.2,1.0)),
            T.RandomHorizontalFlip(),
            T.RnadomApply([T.ColorJitter(0.8,0.8,0.8,0.2)],p=0.8),
            T.RandomGrayScale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=self.im//20*2+1, sigma=(0.1,2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize(*self.norm)
        ])

    def __call__(self, x):
        return self.transform(x), self.tranform(x)

    @classmethod
    def resolve_args(cls, args, eval_stage):
        return cls(args=args)


