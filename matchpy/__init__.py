# -*- coding: utf-8 -*-
"""Contains all the necessary classes and functions for pattern matching."""

# pylint: disable=wildcard-import
from . import expressions
from . import functions
from . import utils
from . import matching

from .expressions import *
from .functions import *
from .utils import *
from .matching import *

__all__ = expressions.__all__ + functions.__all__ + utils.__all__ + matching.__all__
