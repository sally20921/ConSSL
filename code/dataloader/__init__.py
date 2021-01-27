import os
from pathlib import Path

import torch
import torchvision

from inflection import underscore
import inspect

dataset_dict = {}
transform_dict = {}

'''
class.__mro__
a tuple of classes that are considered when looking for base classes during method resolution
'''

def add_to_dict():
    path = Path(os.path.dirname(__file__))
    for p in path.glob('*.py'):
        name = p.stem
        parent = p.parent.stem
        if name != "__init__":
            __import__("{}.{}".format(parent, name))
            module = eval(name)
            for member in dir(module):
                member = getattr(module, member)
                if (inspect.isclass(member) and str(member.__name__).endswith('Transform')):
                    transform_dict[underscore(str(member.__name__))] = member
                elif (inspect.isclass(member) and str(member.__name__).endswith('Dataset')):
                    dataset_dict[underscore(str(member.__name__))] = member 


def get_transform(args, eval_stage):
    tf = transform_dict[args.transform]
    tf = tf.resolve_args(args, eval_stage)
    return tf 

def get_dataset(args, transform, eval_stage):
    ds = dataset_dict[args.dataset]
    ds = ds.resolve_args(transform, eval_stage, args.batch, args.shuffle, args.worker, args.image_path)
    return ds

add_to_dict()


