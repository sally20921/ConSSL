import os
from pathlib import Path

from torch import optim

from inflection import underscore

optim_dict = {}
schd_dict = {}

def add_to_dict():
    path = Path(os.path.dirname(__file__))

    for p in path.glob('*.py'):
        name = p.stem
        parent = p.parent.stem

