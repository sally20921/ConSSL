import os
import inspect
from pathlib import Path

from torch.nn.modules.loss import _Loss

from inflection import underscore

loss_dict = {}

def add_loss():
    path = Path(os.path.dirname(__file__))

    for p in path.glob('*.py'):
        name = p.stem
        parent = p.parent.stem
        if name != "__init__":
            __import__("{}.{}".format(parent, name))
            module = eval(name)
            for member in dir(module):
                member = getattr(module, member)
                loss_dict[underscore(str(member.__name__))] = member

def get_loss(args):
    loss = loss_dict[args.loss_name]
    loss = loss.resolve_args(args)
    return loss.to(args.device)

add_loss()
