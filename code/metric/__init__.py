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


def get_metrics(args):
    metrics = {k: v for k, v in metric_dict.items() if k in args.metrics}
    for k, v in metrics.items():
        if hasattr(v, 'default_transform'):
            metrics[k] = v(output_transform=v.default_transform)
        else:
            metrics[k] = v()
    #k = 'ngram'
    #v = metric_dict[k]
    #if len(set(v.ngrams) & set(args.metrics)) > 0:
    #    metrics[k] = v(args, vocab, v.default_transform)

    return metrics


add_metrics()
