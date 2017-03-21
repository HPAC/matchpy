# -*- coding: utf-8 -*-
"""Contains various patter matching algorithms in the submodules."""

from . import many_to_one
from . import bipartite
from . import one_to_one
from . import syntactic

# pylint: disable=wildcard-import
from .many_to_one import *
from .bipartite import *
from .one_to_one import *
from .syntactic import *

__all__ = many_to_one.__all__ + bipartite.__all__ + one_to_one.__all__ + syntactic.__all__
