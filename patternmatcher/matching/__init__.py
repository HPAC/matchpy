# -*- coding: utf-8 -*-
# pylint: disable=wildcard-import
from .automaton import *
from .bipartite import *
from .many_to_one import *
from .one_to_one import *
from .syntactic import *

import importlib

__all__ = []

for subpackage in ['automaton', 'bipartite', 'many_to_one', 'one_to_one', 'syntactic']:
    package = importlib.import_module('.{}'.format(subpackage), __name__)
    __all__.extend(package.__all__)
