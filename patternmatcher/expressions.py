# -*- coding: utf-8 -*-
import math
from typing import Optional, Any, Tuple, Dict, Callable
from enum import Enum
from collections import Counter as Multiset

# TODO: Is this needed?
# class MatchType(Enum):
#     constant = 1
#     static = 2
#     dynamic = 3

class Arity(tuple, Enum):
    """Arity of an operator as (`int`, `int`) tuple.
    
    First component is the minimum number of operands, second the maximum number.
    Hence, the first component of the tuple must not be larger than the second.
    """
    nullary = (0, 0)
    unary = (1, 1)
    binary = (2, 2)
    ternary = (3, 3)
    polyadic = (2, math.inf)
    variadic = (1, math.inf)

Match = Dict[str, Tuple['Expression']]
Constraint = Callable[[Match], bool]

class Expression(object):
    """Base class for all expressions.
    
    All expressions are immutable, i.e. their attributes should not be changed,
    as several attributes are computed at instantiation and are not refreshed.
    
    Attributes:
        head
            The head of the expression, i.e. the operator for an `Operation` or
            the type for an `Atom`.
        min_count
            For a pattern, this is the minimal count of expression which
            are needed to match (e.g. `x, y` needs at two expressions for a match).
        max_count
            For a pattern, this is the maximal count of expression which
            can be matched. This will either be a constant for patterns with constant
            length or `math.inf` if the pattern contains unbounded wildcards (at top level).
        flexible_match
            `True`, if the expressions contains associative operations
            and hence allows a flexible operand count.
        variables
            Contains a list of all variables in the expression
            with a count of their occurence.
        constraint
            An optional constraint expression, which is checked for each match
            to verify it.
    """

    def __init__(self, head: Any, constraint : Optional[Constraint] = None):
        self._parent = None # type: Optional[Expression]
        self._position = None # type: Optional[int]
        self.head = head
        self.min_count = 1
        self.max_count = 1
        self.flexible_match = False
        self.variables = Multiset() # type: Multiset[str]
        self.constraint = constraint

    # TODO: call it canonicalize maybe?
    def simplify(self):
        """Simplifies the expression using properties like operation associativity or """
        return self

    @property
    def is_constant(self):
        """True, if the expression does not contain any variables."""
        return len(self.variables) == 0

    @property
    def is_static(self):
        """True, if the expression has a fixed width, i.e. it matched a fixed number of expressions."""
        return self.min_count == self.max_count

    @property
    def is_linear(self):
        """True, if the expression is linear, i.e. every variable may occur at most once."""
        return self.is_constant or self.variables.most_common(1)[0][1] == 1

class Operator(object):
    """Base class for all expressions.
    
    All expressions are immutable, i.e. their attributes should not be changed,
    as several attributes are computed at instantiation and are not refreshed.
    
    Attributes:
        name
            Name or symbol for the operator.
        arity
            The arity of the operator as (`int`, `int`) tuple. The first component represents
            the minimum number of operands, the second the maximum number. See the :class:`Arity` enum.
        associative
            The arity of the operator as (`int`, `int`) tuple. The first component represents
            the minimum number of operands, the second the maximum number. See the :class:`Arity` enum.
    """

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
    def __init__(self, operator: Operator, *operands: Expression, constraint:Optional[Constraint]=None) -> None:
        super().__init__(operator)
        self.operator = operator
        self.operands = list(operands)
        self.constraint = constraint

        for i, o in enumerate(operands):
            self.variables.update(o.variables)
            o._parent = self
            o._position = i

        self.flexible_match = operator.associative or any(o.flexible_match for o in operands)

    def __str__(self):
        return '%s(%s)' % (str(self.operator), ', '.join(str(o) for o in self.operands))

class Atom(Expression):
    pass

class Symbol(Atom):
    def __init__(self, name: str) -> None:
        super().__init__(Symbol)
        self.name = name

    def __str__(self):
        return self.name

class Wildcard(Atom):
    def __init__(self, name: str) -> None:
        super().__init__(Symbol)
        self.name = name
        self.variables.update([name])

    def __str__(self):
        return self.name + '_'

if __name__ == '__main__':
    f = Operator('f', Arity.binary)
    x = Wildcard('x')
    y = Wildcard('y')

    expr = Operation(f, x, y)

    print(expr.is_linear)
