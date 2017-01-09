# -*- coding: utf-8 -*-
# pylint: disable=wildcard-import
from .many_to_one import *
from .bipartite import *
from .one_to_one import *
from .syntactic import *

import importlib

__all__ = []

for subpackage in ['many_to_one', 'bipartite', 'one_to_one', 'syntactic']:
    package = importlib.import_module('.{}'.format(subpackage), __name__)
    __all__.extend(package.__all__)
