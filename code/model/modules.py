from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock
#from torchvision.models import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import copy
import random
from functools import wraps

# get resnet models

def get_resnet(resnet):
    if resnet == 'resnet18':
        return ResNet(BasicBlock, [2,2,2,2])
    elif resnet == 'resnet34':
        return ResNet(BasicBlock, [3,4,6,3])
    elif resnet == 'resnet50':
        return ResNet(Bottleneck, [3,4,6,3])
    elif resnet == 'resnet101':
        return ResNet(Bottleneck, [3,4,23,3])
    elif resnet == 'resnet152':
        return ResNet(Bottleneck, [3,8,36,3])


class Projection(nn.Module):
    def __init__(self, in_dim, out_dim=2048, hsz=2048, n_layers=3):
        super().__init__()

        layers = []
        prev_dim = in_dim
        for i in range(n_layers):
            if i == n_layers - 1:
                layers.extend([
                    nn.Linear(prev_dim, out_dim),
                    nn.BatchNorm1d(out_dim)
                ])
            else:
                layers.extend([
                    nn.Linear(prev_diim, hsz),
                    nn.BatchNorm1d(hsz),
                    nn.ReLU(inplace=True)
                ])
                prev_dim = hsz
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class Prediction(nn.Module):
    def __init__(self, in_dim=2048, out_dim=2048, hsz=512, n_layers=2):
        super().__init__()

        layers = []
        prev_dim = in_dim
        for i in range(n_layers):
            if i == n_layers - 1:
                layers.append(nn.Linear(prev_dim, out_dim))
            else:
                layers.extend([
                    nn.Linear(prev_dim, hsz),
                    nn.BatchNorm1d(hsz),
                    nn.ReLU(inplace=True)
                ])
                prev_dim = hsz

            self.main = nn.Sequential(*layers)

        def forward(self, x):
            return self.main(x)
