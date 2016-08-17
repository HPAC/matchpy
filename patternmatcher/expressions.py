# -*- coding: utf-8 -*-
import math
from collections import Counter as Multiset
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

from patternmatcher.utils import isidentifier


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
        variables
            Contains a list of all variables in the expression
            with a count of their occurence.
        constraint
            An optional constraint expression, which is checked for each match
            to verify it.
    """

    def __init__(self, constraint : Optional[Constraint] = None):
        self.min_count = 1
        self.max_count = 1
        self.variables = Multiset() # type: Multiset[str]
        self.constraint = constraint

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
        if self.is_constant:
            return True
        common = self.variables.most_common(2)
        if common[0][1] == 1:
            return True
        if common[0][0] == '_' and (len(common) == 1 or common[1][1] == 1):
            return True
        return False

    def preorder_iter(self, predicate=None):
        """Iterates over all subexpressions that match the (optional) `predicate`."""
        if filter is None or predicate(self):
            yield self

class Operation(Expression):
    """Base class for all operations."""

    name = None # type: str
    """Name or symbol for the operator."""

    arity = Arity.variadic # type: Tuple[int, int]
    """The arity of the operator as (`int`, `int`) tuple.

    The first component represents
    the minimum number of operands, the second the maximum number. See the :class:`Arity` enum.
    """

    associative = False
    """True if the operation is associative, i.e. `f(a, f(b, c)) = f(f(a, b), c)`.
    
    This property is used to flatten nested associative operations of the same type.
    Therefore, the `arity` of an associative operation has to have an unconstraint maximum
    number of operand. 
    """

    commutative = False
    """True if the operation is commutative, i.e. `f(a, b) = f(b, a)`.
    
    Note that commutative operations will always be converted into canonical
    form with sorted operands.
    """

    one_identity = False
    """True if the operation with a single argument is equivalent to the identity function.
    
    This property is used to simplify expressions, e.g. for `f` with `f.one_identity = True`
    the expression `f(a)` if simplified to `a`.
    """

    def __new__(cls, *operands: Expression, constraint:Optional[Constraint]=None):
        """Base class for all expressions.

        All expressions are immutable, i.e. their attributes should not be changed,
        as several attributes are computed at instantiation and are not refreshed.

        Arguments:
            operands
                The operands for the operation expression.
            constraint
                An optional constraint expression, which is checked for each match
                to verify it.
        """
        operation = object.__new__(cls)
        Expression.__init__(operation, constraint)

        operation.operands = list(operands)
        operation = cls._simplify(operation)

        if type(operation) == cls:
            for i, o in enumerate(operands):
                operation.variables.update(o.variables)

        return operation

    def __init__(self, *operands: Expression, constraint:Optional[Constraint]=None): # pylint: disable=W0231
        # Expression.__init__ is called in __new__()
        pass

    def __str__(self):
        if self.constraint:
            return '%s(%s) /; %s' % (self.name, ', '.join(str(o) for o in self.operands), str(self.constraint)) 
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

    def __lt__(self, other):
        if isinstance(other, Symbol):
            return False

        if type(self) != type(other):
            return type(self).__name__ < type(other).__name__

        if len(self.operands) != len(other.operands):
            return len(self.operands) < len(other.operands)
        
        for left, right in zip(self.operands, other.operands):
            if left < right:
                return True

        return False

    def __eq__(self, other):
        return type(self) == type(other) and \
               self.constraint == other.constraint and \
               len(self.operands) == len(other.operands) and \
               all(x == y for x,y in zip(self.operands, other.operands))

    @staticmethod
    def _simplify(operation: 'Operation') -> Expression:        
        if operation.associative:
            newOperands = []
            for operand in operation.operands:
                if type(operand) == type(operation):
                    newOperands.extend(operand.operands)
                else:
                    newOperands.append(operand)
            operation.operands = newOperands
        
        if operation.one_identity and len(operation.operands) == 1:
            return operation.operands[0]

        if operation.commutative:
            operation.operands.sort()
        
        return operation

    def preorder_iter(self, predicate=None):
        """Iterates over all subexpressions that match the (optional) `predicate`."""
        yield from super().preorder_iter(predicate)
        for operand in self.operands:
            yield from operand.preorder_iter(predicate)

class Atom(Expression):
    pass

class Symbol(Atom):
    def __init__(self, name: str, constraint:Optional[Constraint]=None) -> None:
        super().__init__(constraint)
        self.name = name

    def __str__(self):
        return self.name

    def __lt__(self, other):
        if isinstance(other, Symbol):
            return self.name < other.name
        return True

class Variable(Atom):
    """A variable that is captured during a match.

    Wraps another pattern expression that is used to match. On match, the matched
    value is captured for the variable.
    """

    def __init__(self, name: str, expression: Expression, constraint:Optional[Constraint]=None) -> None:
        """
        Arguments:
            name
                The name of the variable that is used to capture its value in the match.
                Can be used to access its value in constraints or for replacement.
            expression
                The expression that is used for matching. On match, its value will be
                assigned to the variable. Usually, a :class:`Wildcard` is used to match
                any expression.
            constraint
                See :class:`Expression`.
        """
        super().__init__(constraint)
        self.name = name
        self.expression = expression
        self.variables.update([name])
        self.min_count = expression.min_count
        self.max_count = expression.max_count

    @staticmethod
    def dot(name: str, constraint:Optional[Constraint]=None):
        """Creates a `Variable` with a :class:`Wildcard` that matches exactly one argument."""
        return Variable(name, Wildcard.dot(), constraint)

    @staticmethod
    def star(name: str, constraint:Optional[Constraint]=None):
        """Creates a `Variable` with :class:`Wildcard` that matches any number of arguments."""
        return Variable(name, Wildcard.star(), constraint)

    @staticmethod
    def plus(name: str, constraint:Optional[Constraint]=None):
        """Creates a `Variable` with :class:`Wildcard` that matches at least one and up to any number of arguments."""
        return Variable(name, Wildcard.plus(), constraint)

    def __str__(self):
        if isinstance(self.expression, Wildcard):
            value = self.name + str(self.expression)
        else:
            value = '%s_: %s' % (self.name, self.expression)
        if self.constraint:
            value += ' /; %s' % str(self.constraint)

        return value

    def __eq__(self, other):
        return isinstance(other, Variable) and self.name == other.name and self.expression == other.expression

    def __lt__(self, other):
        if isinstance(other, Symbol):
            return False
        if isinstance(other, Variable):
            return self.name < other.name
        return type(self).__name__ < type(other).__name__

class Wildcard(Atom):
    """A wildcard that matches any expression.
    
    The wildcard will match any number of expressions between `min_count` and `max_count`.
    Optionally, the wildcard can also be constrained to only match expressions satisfying a predicate.
    """

    def __init__(self, min_count: int, max_count: int, constraint: Optional[Constraint]=None) -> None:
        """
        Arguments:
            min_count
                The minimum number of expressions this wildcard will match. Must not be
                greater than `max_count` or negative.
            max_count
                The maximum number of expressions this wildcard will match. Use `math.inf`
                for an unbounded wildcard. Must not be smaller than `min_count`.
            constraint
                An optional constraint for expressions to be considered a match. If set, this
                callback is invoked for every match and the return value is utilized to decide
                whether the match is valid.
        """
        assert min_count >= 0, 'Negative min_count'
        assert min_count <= max_count, 'min_count > max_count'

        super().__init__(constraint)
        self.variables.update(['_'])
        self.min_count = min_count
        self.max_count = max_count

    @staticmethod
    def dot():
        """Creates a :class:`Wildcard` that matches exactly one argument."""
        return Wildcard(min_count=1, max_count=1)

    @staticmethod
    def star():
        """Creates a :class:`Wildcard` that matches any number of arguments."""
        return Wildcard(min_count=0, max_count=math.inf)

    @staticmethod
    def plus():
        """Creates a :class:`Wildcard` that matches at least one and up to any number of arguments."""
        return Wildcard(min_count=1, max_count=math.inf)

    def __str__(self):
        if self.max_count == math.inf:
            if self.min_count == 0:
                return '___'
            elif self.min_count == 1:
                return '__'
        return '_'

    def __lt__(self, other):
        return isinstance(other, Wildcard)

    def __eq__(self, other):
        return isinstance(other, Wildcard) and \
               other.min_count == self.min_count and \
               other.max_count == self.max_count

if __name__ == '__main__':
    f = Operation.new('f', Arity.binary, associative = True, commutative=True, one_identity=True)
    x = Variable.dot('x')
    y = Variable.star('y')
    z = Variable.plus('z')
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')

    expr = f(f(b), a)

    print(expr)