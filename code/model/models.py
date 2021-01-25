import torch
import torch.nn as nn
import torch.nn.functional as F

from math import *
import random
from copy import ddepcopy

from .modules import * # resnet, prediction, projection class 
from collections import defaultdict

class SimSiam(nn.Module):
    def __init__(self, resnet, use_outputs):
        super(SimSiam, self).__init__()

        self.backbone = resnet
        self.projector = Projection(self.backbone.output_dim)
        self.predictor = Prediction()
        self.encoder = nn.Sequential(self.backbone, self.projector)

        self.net_output_key = use_outputs

    def forward(self, x_i, x_j):
        f, h = self.encoder, self.predictor
        
        z_i, z_j = f(x_i), f(x_j)
        p_i, p_j = h(z_i), h(z_j)
        y_pred = { key : eval(key) for key in self.net_output_key }

        return y_pred

    @classmethod
    def resolve_args(cls, args):
        resnet = get_resnet[args.resnet]
        return cls(args, resnet, args.use_outputs)

class BYOL(nn.Module):
    def __init__(self, args, resnet, use_outputs, base_momentum=0.996):
        super().__init__()

        self.t = base_momentum
        self.backbone = resnet
        self.projector = Projection(resnet50.output_dim, 256, 4096)
        #self.encoder = nn.Sequential(self.backbone, self.projector)
        self.predictor = Predictor(256, 256, 4096)

        self.online_network =  nn.Sequential(self.backbone, self.projector)
        self.target_network = nn.Sequential(self.backbone, self.projector)
        self._initailize()

        self.net_output_key = use_outputs

    def resolve_args(cls, args):
        resnet = get_resnet[args.resnet]
        return cls(args, resnet, args.use_outputs, args.base_momentum)

    @torch.no_grad()
    def _initialize(self):
        for p, q in zip(self.online_network.parameters(), self.target_network.parameters()):
            q.data.copy_(p.data)
            q.requires_grad = False

    @torch.no_grad()
    def _update(self, t):
        '''momentum update of target network'''
        for p, q in zip(self.online_network.parameters(), self.target_network.parameters()):
            q.data.mul_(t).add_(1-t, p.data)

    def forward(self, x_i, x_j):
        # online network forward
        p = self.predictor(self.online_network(x_i, x_j))

        # target network forward
        with torch.no_grad():
            self._update(self.t)
            z = self.target_network(x_i, x_j)
            {key: eval(key) for key in self.net_output_key}

            return y_pred
        








        
        
        

