import math
import os
from collections import defaultdict

import PIL
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import datasets, transforms
from utils import *

from .vision import VisionDataset

image_types = ['cifar10', 'cifar100', 'imagenet']
transform_types = ['sim_siam', 'train_eval', 'eval']
# image_size = # imagenet 224 # cifar 32
delimiter = '/'
# two_crops_transform = True

'''
torchvision.datasets are subclasses of torch.utils.data.Dataset.
i.e. they have __getitem__ and __len__ methods implemented.
Hence, they can all be passed to torch.utils.data.DataLoader.
'''

'''
torchvision.datasets.ImageFolder(root: str, transform: Optional[Callable]=None,
target_transform: Optional[Callable]=None, ...)
Args:
    transform (callable, optional) - A function/transform that takes in an PIL image and returns a transformed version.
    target_transform (callable, optional) - A function/transform that takes in the target and transforms it.
'''
#----------------------CIFAR----------------------------------------#
mean_std = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

cifar_transform_dict = {
        'eval': transforms.Compose([
            transforms.ToTensor(),
            mean_std
            ]),
        'train_eval': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            mean_std
            ])
        }
#--------------------IMAGENET---------------------------------------#
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imagenet norm

imagenet_transform_dict = {
        'sim_siam': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0,2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            T.RandomApply([transforms.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            normalize
            ]), # similar to simclr, mocov2
        'train_eval': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RnadomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]),
        'eval': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
            ])
        }
#------------------------------------------------------------------------#
class TwoCropsTransform: # only for train
    ''' takes two random crops of one image as the query and key '''
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        q = self.transform(x)
        k = self.transform(x)
        return q, k

# ------------------------------------------------------------------------#
def preprocess_images(args):
    if args.datasets == 'imagenet':
        return load_imagenet(args)
    else:
        return load_cifar(args)

# --------------------------------------------------------------------------#
def load_imagenet(args):
    traindir = os.path.join(args.image_path, 'train')
    valdir = os.path.join(args.image_path, 'val')

    trainloader = torch.utils.data.DataLoader(datasets.ImageFolder(traindir, TwoCropsTransform(imagenet_transform_dict[args.model_name])), batch_size=args.batch_sizes[0], shuffle=args.shuffle[0], num_workers=args.num_workers, pin_memory=True)

    traintestloader = torch.utils.data.DataLoader(datasets.ImageFolder(traindir, imagenet_transform_dict['train_eval']), batch_size=args.batch_sizes[0], shuffle=args.shuffle[0], num_workers=args.num_workers, pin_memory=True)


    valloader = torch.utils.data.DataLoader(datasets.ImageFolder(valdir, imagenet_transform_dict['eval']), batch_size=args.batch_sizes[1], shuffle=args.shuffle[1], num_workers=args.num_workers, pin_memory=True)
    print('preparing imagenet dataset completed')
    return trainloader, traintestloader, valloader

def load_cifar(args):
    if args.datasets == 'cifar10':
        dataloader = datasets.CIFAR10
    else:
        dataloader = datasets.CIFAR100

    trainset = dataloader(root=args.image_path, train=True, download=True, transform=TwoCropsTransform(cifar_transform_dict['train_eval']))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_sizes[0], shuffle=args.shuffle[0], num_workers=args.num_workers)

    traintestset = dataloader(root=args.image_path, train=True, download=True, transform=cifar_transform_dict['train_eval'])
    traintestloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_sizes[0], shuffle=args.shuffle[0], num_workers=args.num_workers)

    testset = dataloader(root=args.image_path, train=False, download=True, transform=cifar_transform_dict['eval'])
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_sizes[1], shuffle=args.shuffle[1], num_workers=args.num_workers)

    print('preparing dataset cifar completed')

    return trainloader, traintestloader, testloader



