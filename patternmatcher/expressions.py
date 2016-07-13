# -*- coding: utf-8 -*-
import sys
from typing import Optional, List, Any, Tuple
from enum import Enum

class MatchType(Enum):
    constant = 1
    static = 2
    dynamic = 3

class Arity(tuple, Enum):
    nullary = (0, 0)
    unary = (1, 1)
    binary = (2, 2)
    ternary = (3, 3)
    polyadic = (2, sys.maxsize)
    variadic = (1, sys.maxsize)

class Expression(object):
    def __init__(self, min_count, max_count):
        self.parent = None # type: Optional[Expression]
        self.position = None # type: Optional[int]
        self.head = None # type: Optional[Any]
        self.min_count = min_count
        self.max_count = max_count
        self.match_type = MatchType.dynamic

    def simplify(self):
        return self

class Operator(object):
    def __init__(self, name: str, arity: Tuple[int, int], associative: bool = False, commutative: bool = False, oneIdentity: bool = False, neutralElement: Optional[Expression] = None) -> None:
        self.name = name
        self.arity = arity
        self.associative = associative
        self.commutative = commutative
        self.oneIdentity = oneIdentity
        self.neutralElement = neutralElement
        
    def __str__(self):
        return self.name

class Operation(Expression):
    pass

class Atom(Expression):
    pass

class Symbol(Atom):
    pass

class Wildcard(Atom):
    pass
