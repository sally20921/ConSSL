import torch
import torch.nn as nn
import torch.nn.functional as F

from math import *
import random
from copy import deepcopy

from .modules import * # resnet, prediction, projection class 
from collections import defaultdict
from torchvision.models import resnet50

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(ProjectionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l3 = nn.Sequential(
            nn.Linear(mid_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x


class PredictionMLP(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(PredictionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)

        return x
    
class SimSiam(nn.Module):
    def __init__(self, resnet, use_outputs):
        super(SimSiam, self).__init__()

        self.net = resnet50()
        #self.projector = Projection(2048)
        #self.predictor = Prediction()
        #self.encoder = nn.Sequential(self.backbone, self.projector)
        num_ftrs = self.net.fc.in_features
        self.features = nn.Sequential(*list(self.net.children())[:-1])

        self.projection = ProjectionMLP(num_ftrs, 2048, 2048)
        self.prediction = PredictionMLP(2048, 512, 2048)
        
        self.net_output_key = use_outputs
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # reset conv initialization to default uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                stdv = 1. / math.sqrt(n)
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)
    def forward(self, x):
        #f, h = self.encoder, self.predictor
        
        #z_i, z_j = f(x_i), f(x_j)
        #p_i, p_j = h(z_i), h(z_j)
        #y_pred = { key : eval(key) for key in self.net_output_key }
        x = self.features(x)
        x = x.view(x.size(0), -1)
        z = self.projection(x)
        p = self.prediction(z)
        y_pred = {'z': z, 'p': p}
        return y_pred

    @classmethod
    def resolve_args(cls, args):
        #resnet = get_resnet[args.resnet]
        return cls(args.resnet, args.use_outputs)

class BYOL(nn.Module):
    def __init__(self, args, resnet, use_outputs, base_momentum=0.996):
        super().__init__()

        self.t = base_momentum
        self.backbone = resnet50()
        self.projector = Projection(2048, 256, 4096)
        #self.encoder = nn.Sequential(self.backbone, self.projector)
        self.predictor = Predictor(256, 256, 4096)

        self.online_network =  nn.Sequential(self.backbone, self.projector)
        self.target_network = nn.Sequential(self.backbone, self.projector)
        self._initialize()

        self.net_output_key = use_outputs

    def resolve_args(cls, args):
        #resnet = get_resnet[args.resnet]
        return cls(args, args.resnet, args.use_outputs, args.base_momentum)

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
        








        
        
        

