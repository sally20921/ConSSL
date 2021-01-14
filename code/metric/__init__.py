import os
from pathlib import Path

from torch import optim
from ignite.metrics.metric import Metric

from inflection import underscore

metric_dict = {}

def add_metrics():
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
                        Metric in member.__bases__:
                            metric_dict[underscore(str(member.__name__))] = member

add_metrics()

