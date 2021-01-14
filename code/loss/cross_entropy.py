import math

from torch import nn
from torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from metric.stat_metric import StatMetric

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, eps=0, padding_idx=0):
        self.eps = eps
        #self.padding_idx = padding_idx

        #super(CrossEntropyLoss,self).__init__(ignore_index=padding_index)
        super(CrossEntropyLoss, self).__init__()

    # static method: bound to a class 
    # can be called without an object 
    @staticmethod
    def get_metric():
        # negative log likelihood loss 
        return {'nll_loss': StatMetric(output_transform=lambda x: (x[1]['nll_loss'], x[2]))}
        # return loss.item(), stats, batch_size, y_pred.detach(), target.detach()

    def forward(self, hypo, tgt):
        hypo = hypo.contiguous()
        tgt = tgt.contiguous()
        
        # failsafe for empty tensor
        if hypo.nelement() == 0 or tgt.nelement() == 0:
            loss = None
        else:
            loss = super().forward(hypo.view(-1, hypo.shape[-1]), tgt.view(-1))
        return loss, {'nll_loss': loss.item()}

    def _reduce(self, t):
        func = {
                'none': lambda x: x,
                'mean': lambda x: x.mean(),
                'sum': lambda x: x.sum()
        }[self.reduction]

        return func(t)

    @classmethod
    def resolve_args(cls, args):
        eps = args.get("label_smoothing", 0)
        return cls(eps=eps)
