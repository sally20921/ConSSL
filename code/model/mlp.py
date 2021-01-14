import torch
from torch import nn, utils
from torchvision import models, datasets, transforms
from utils import *

def get_model(args):
    print('Loading extractor model: using resnet18')

    model = models.resnet18(pretrained=True)
    extractor = nn.Sequential(*list(model.children())[:-2])
    extractor.to(args.device)

    return extractor
