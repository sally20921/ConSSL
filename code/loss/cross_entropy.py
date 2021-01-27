from torch import nn

class CrossEntropyLoss(nn.CrossEntropyLoss):

    @classmethod
    def resolve_args(args):
        return cls()
