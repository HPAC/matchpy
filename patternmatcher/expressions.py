# -*- coding: utf-8 -*-
import math
from typing import Optional, Any, Tuple, Dict, Callable
from enum import Enum
from collections import Counter as Multiset
from .utils import isidentifier

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

    def __init__(self, constraint : Optional[Constraint] = None):
        self._parent = None # type: Optional[Expression]
        self._position = None # type: Optional[int]
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

class Operation(Expression):
    """Base class for all operations.

    Attributes:
        name
            Name or symbol for the operator.
        arity
            The arity of the operator as (`int`, `int`) tuple. The first component represents
            the minimum number of operands, the second the maximum number. See the :class:`Arity` enum.
        associative
            True if the operation is associative, i.e. `f(a, f(b, c)) = f(f(a, b), c)`.
            This property is used to flatten nested associative operations of the same type.
            Therefore, the `arity` of an associative operation has to have an unconstraint maximum
            number of operand. 
        commutative
            True if the operation is commutative, i.e. `f(a, b) = f(b, a)`. Note that commutative
            operations will always be converted into canonical form with sorted operands.
        one_identity
            True if the operation with a single argument is equivalent to the identity function, i.e.
            `f(a) = a`. This property is used to simplify expressions.
    """
    name = None # type: str
    arity = Arity.variadic # type: Tuple[int, int]
    associative = False
    commutative = False
    one_identity = False

    def __init__(self, *operands: Expression, constraint:Optional[Constraint]=None) -> None:
        super().__init__(constraint)
        self.operands = list(operands)
        self.constraint = constraint

        for i, o in enumerate(operands):
            self.variables.update(o.variables)
            o._parent = self
            o._position = i

        self.flexible_match = self.associative or any(o.flexible_match for o in operands)

    def __str__(self):
        return '%s(%s)' % (self.name, ', '.join(str(o) for o in self.operands))

    @staticmethod
    def new(name : str, arity : Tuple[int, int], class_name : str = None, **attributes) -> Any:
        """Utility method to create a new operation type.

        Example:

        >>> Times = Operator.new('*', Arity.polyadic, 'Times', associative=True, commutative=True, one_identity=True)
        >>> Times
        <class 'patternmatcher.expressions.Times'>
        >>> str(Times(Symbol('a')))
        '*(a)'

        Arguments:
            name
                Name or symbol for the operator. Will be used as name for the new class if
                `class_name` is not specified.
            arity
                The arity of the operator as explained in the documentation of :class:`Operation`.
            class_name
                Name for the new operation class to be used instead of name. This argument
                is required if `name` is not a valid python identifier.
            attributes
                Attributes to set in the new class. For a list of possible attributes see the
                docstring of :class:`Operation`.
        """
        class_name = class_name or name
        assert isidentifier(class_name), 'invalid identifier'

        return type(class_name, (Operation,), dict({
            'name': name,
            'arity': arity
        }, **attributes))

class Atom(Expression):
    pass

class Symbol(Atom):
    def __init__(self, name: str, constraint:Optional[Constraint]=None) -> None:
        super().__init__(constraint)
        self.name = name

    def __str__(self):
        return self.name

class Variable(Atom):
    def __init__(self, name: str, expression: Expression, constraint:Optional[Constraint]=None) -> None:
        super().__init__(constraint)
        self.name = name
        self.expression = expression
        self.variables.update([name])

    @staticmethod
    def dot(name: str, constraint:Optional[Constraint]=None):
        return Variable(name, Wildcard.dot(), constraint)

    @staticmethod
    def star(name: str, constraint:Optional[Constraint]=None):
        return Variable(name, Wildcard.star(), constraint)

    @staticmethod
    def plus(name: str, constraint:Optional[Constraint]=None):
        return Variable(name, Wildcard.plus(), constraint)

    def __str__(self):
        if isinstance(self.expression, Wildcard):
            return self.name + str(self.expression)

        return '%s_: %s' % (self.name, self.expression)

class Wildcard(Atom):
    def __init__(self, min_count: int, max_count: int, constraint: Optional[Constraint]=None) -> None:
        super().__init__(constraint)
        self.min_count = min_count
        self.max_count = max_count

    @staticmethod
    def dot():
        return Wildcard(min_count=1, max_count=1)

    @staticmethod
    def star():
        return Wildcard(min_count=0, max_count=math.inf)

    @staticmethod
    def plus():
        return Wildcard(min_count=1, max_count=math.inf)

    def __str__(self):
        if self.min_count == 0 and self.max_count == math.inf:
            return '___'
        if self.min_count == 1 and self.max_count == math.inf:
            return '__'
        return '_'

if __name__ == '__main__':
    f = Operation.new('f', Arity.binary, associative = False)
    x = Variable.dot('x')
    y = Variable.star('y')

    expr = f(x, y)

    print(expr)