import os
from pathlib import Path

from torch import nn
from inflection import underscore

model_dict = {}
linear_dict = {}

def add_models():
    path = Path(os.path.dirname(__file__))

    for p in path.glob('*.py'):
        name = p.stem
        parent = p.parent.stem
        if name != "__init__":
            __import__("{}.{}".format(parent, name))
            module = eval(name)
            for member in dir(module):
                member = getattr(module, member)
                if hasattr(member, "__bases__") and \
                        nn.Module in member.__bases__:
                    model_dict[underscore(str(member.__name__))] = member
                    linear_dict[underscore(str(member.__name__))] = member 

def get_model(args):
    model = model_dict[args.model_name]
    model = model.resolve_args(args)
    return model.to(args.device)

def get_linear(args, pt_model, num_classe):
    model = linear_dict[args.model_name]
    model = model(args, pt_model, num_classes)
    return model.to(args.device)

add_models()

