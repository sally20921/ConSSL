import os
import inspect
from pathlib import Path

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
                if (inspect.isclass(member) and str(member.__name__).endswith('Loss')):
                    loss_dict[underscore(str(member.__name__))] = member
