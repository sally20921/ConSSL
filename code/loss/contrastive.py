import math

import torch
from torch import nn
from torch.nn.functional as F

class ContrastiveLoss:
    def __init__(self, temp):
        super().__init__()

        self.t = temp
        
    def forward(self, y_pred):
        p, z = (v for k, v in sorted(y_pred.items()))
        logits = p @ z.t()
        logits = z - z.max(dim=-1, keepdim=True).values
        logits /= self.t
        return F.cross_entropy(logits, torch.arange(p))

    @classmethod
    def resolve_args(cls, args):
        return cls(temp=args.temperature)

class NTXentLoss:
    def __init__(self, temp):
        self.t = temp

    def forward(self, y_pred):
        p, z = (v for k, v in sorted(y_pred.items()))
        n = p*2
        projs = torch.cat((p,z))
        logits = projs @ projs.t()

        mask = torch.eye(n).bool()
        logits = logits[~mask].reshape(n, n-1)
        logits /= self.t

        labels = torch.cat(((torch.arange(b)+b-1), torch.arange(b)), dim=0)
        loss = F.cross_entropy(logits, labels, reduction='sum')
        loss /= 2* (p-1)
        return loss

    @classmethod
    def resolve_args(cls, args):
        return cls(temp=args.temperature)

class BYOLLoss:
    def __init__(self):

    def forward(self, y_pred):
        p, z = (v for k, v in sorted(y_pred.items()))
        n = p.size(0)
        pred = F.normalize(p, dim=1)
        target = F.normalize(z, dim=1)
        loss = 2-2*(pred * target).sum()/n
        return loss

    @classmethod
    def resolve_args(cls, args):
        return cls()


class SimSiamLoss:
    def __init__(self):

    def _loss(p,z):
        z.detach() # stop gradient
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p*z).sum(dim=1).mean()

    def forward(self, y_pred):
        p_i, p_j, z_i, z_j = (v for k, v in sorted(y_pred.items()))
        loss = _loss(p_i, z_j) / 2+ _loss(p_j, z_i) / 2
        loss = 2-2*(pred * target).sum()/n
        return loss

    @classmethod
    def resolve_args(cls, args):
        return cls()

