# -*- coding: utf-8 -*-
import keyword
from enum import Enum, EnumMeta
from typing import (Callable, Dict, Iterator, List, NamedTuple, Optional,
                    Set, Tuple, TupleMeta, Type, Union)

from patternmatcher.multiset import Multiset


# This class is needed so that Tuple and Enum play nicely with each other
class _ArityMeta(TupleMeta, EnumMeta):
    @classmethod
    def __prepare__(metacls, cls, bases, **kwargs):
        return super().__prepare__(cls, bases)

_ArityBase = NamedTuple('_ArityBase', [('min_count', int), ('fixed_size', bool)])

class Arity(_ArityBase, Enum, metaclass=_ArityMeta, _root=True):
    """Arity of an operator as (`int`, `bool`) tuple.

    The first component is the minimum number of operands.
    If the second component is `True`, the operator has fixed width arity. In that case, the first component
    describes the fixed number of operands required.
    If it is `False`, the operator has variable width arity.
    """
    
    def __new__(cls, *data):
        return tuple.__new__(cls, data)

    nullary     = (0, True)
    unary       = (1, True)
    binary      = (2, True)
    ternary     = (3, True)
    polyadic    = (2, False)
    variadic    = (1, False)

    def __repr__(self):
        return "%s.%s" % (self.__class__.__name__, self._name_)

Match = Dict[str, Union['Expression', List['Expression']]]
Constraint = Callable[[Match], bool]
ExpressionPredicate = Callable[['Expression'], bool]

class Expression(object):
    """Base class for all expressions.

    All expressions are immutable, i.e. their attributes should not be changed,
    as several attributes are computed at instantiation and are not refreshed.

    Attributes:
        constraint
            An optional constraint expression, which is checked for each match
            to verify it.
    """

    def __init__(self, constraint: Optional[Constraint] = None) -> None:
        self.constraint = constraint
        self.head = None # type: Union[type,Atom]

    @property
    def variables(self) -> Multiset:
        """"""
        return Multiset()

    @property
    def symbols(self) -> Multiset:
        return Multiset()

    @property
    def is_constant(self) -> bool:
        """True, if the expression does not contain any wildcards."""
        return True

    @property
    def is_syntactic(self) -> bool:
        """True, if the expression does not contain any associative or commutative operations or sequence wildcards."""
        return True

    @property
    def is_linear(self) -> bool:
        """True, if the expression is linear, i.e. every variable may occur at most once."""
        return self._is_linear(set())

    def _is_linear(self, variables: Set[str]) -> bool:
        return True

    def preorder_iter(self, predicate:Optional[ExpressionPredicate]=None, position:Tuple[int,...]=()) -> Iterator[Tuple['Expression',Tuple[int,...]]]:
        """Iterates over all subexpressions that match the (optional) `predicate`."""
        if filter is None or predicate(self):
            yield self, position

    def __getitem__(self, key):
        if len(key) == 0:
            return self
        raise IndexError("Invalid position")

    def __hash__(self):
        raise NotImplementedError()

class OperationMeta(type):
    def __repr__(cls):
        return 'Operation[%r, arity=%r, associative=%r, commutative=%r, one_identity=%r]' % \
            (cls.name, cls.arity, cls.associative, cls.commutative, cls.one_identity)


class Operation(Expression, metaclass=OperationMeta):
    """Base class for all operations."""

    name = None # type: str
    """Name or symbol for the operator."""

    arity = Arity.variadic # type: Arity
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
        operation.head = cls
        operation = cls._simplify(operation)

        return operation

    def __init__(self, *operands: Expression, constraint: Optional[Constraint]=None) -> None: # pylint: disable=W0231
        # Expression.__init__ is called in __new__()
        # for mypy so that it knows there is a property `operands`
        self.operands = self.operands # type: List[Expression]

    def __str__(self):
        if self.constraint:
            return '%s(%s) /; %s' % (self.name, ', '.join(str(o) for o in self.operands), str(self.constraint))
        return '%s(%s)' % (self.name, ', '.join(str(o) for o in self.operands))

    def __repr__(self):
        operand_str = ', '.join(map(repr, self.operands))
        if self.constraint:
            return '%s(%s, constraint=%r)' % (self.__class__.__name__, operand_str, self.constraint)
        return '%s(%s)' % (self.__class__.__name__, operand_str)

    @staticmethod
    def new(name : str, arity : Arity, class_name : str = None, **attributes) -> Type['Operation']:
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
        if not class_name.isidentifier() or keyword.iskeyword(class_name):
            raise ValueError("Invalid identifier for new operator class.")

        return type(class_name, (Operation,), dict({
            'name': name,
            'arity': arity
        }, **attributes))

    def __lt__(self, other):
        if isinstance(other, Symbol):
            return False

        if not isinstance(other, self.__class__):
            return type(self).__name__ < type(other).__name__

        if len(self.operands) != len(other.operands):
            return len(self.operands) < len(other.operands)

        for left, right in zip(self.operands, other.operands):
            if left < right:
                return True
            elif right < left:
                return False

        return False

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
               self.constraint == other.constraint and \
               len(self.operands) == len(other.operands) and \
               all(x == y for x,y in zip(self.operands, other.operands))

    def __getitem__(self, key):
        if len(key) == 0:
            return self
        head, *remainder = key
        return self.operands[head][remainder]

    @staticmethod
    def _simplify(operation: 'Operation') -> Expression:
        if operation.associative:
            new_operands = [] # type: List[Expression]
            for operand in operation.operands:
                if isinstance(operand, operation.__class__):
                    new_operands.extend(operand.operands) # type: ignore
                else:
                    new_operands.append(operand)
            operation.operands = new_operands

        if operation.one_identity and len(operation.operands) == 1:
            return operation.operands[0]

        if operation.commutative:
            operation.operands.sort()

        return operation

    @property
    def is_constant(self):
        return all(x.is_constant for x in self.operands)

    @property
    def is_syntactic(self):
        if self.associative or self.commutative:
            return False
        return all(o.is_syntactic for o in self.operands)

    @property
    def variables(self):
        return sum((x.variables for x in self.operands), Multiset())

    @property
    def symbols(self):
        return sum((x.symbols for x in self.operands), Multiset([self.name]))

    def _is_linear(self, variables):
        return all(o._is_linear(variables) for o in self.operands)

    def preorder_iter(self, predicate:Optional[ExpressionPredicate]=None, position:Tuple[int,...]=()) -> Iterator[Tuple['Expression',Tuple[int,...]]]:
        if predicate is None or predicate(self):
            yield self, position
        for i, operand in enumerate(self.operands):
            yield from operand.preorder_iter(predicate, position + (i, ))

    def __hash__(self):
        return hash(tuple([type(self)] + self.operands))

class Atom(Expression): # pylint: disable=abstract-method
    pass

class Symbol(Atom):
    def __init__(self, name: str, constraint:Optional[Constraint]=None) -> None:
        super().__init__(constraint)
        self.name = name
        self.head = self

    def __str__(self):
        return self.name

    def __repr__(self):
        if self.constraint:
            return '%s(%r, constraint=%r)' % (self.__class__.__name__, self.name, self.constraint)
        return '%s(%r)' % (self.__class__.__name__, self.name)

    @property
    def symbols(self):
        return Multiset([self.name])

    def __lt__(self, other):
        if isinstance(other, Symbol):
            return self.name < other.name
        return True

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name

    def __hash__(self):
        return hash((type(self), self.name))

class Variable(Expression):
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
        self.head = expression.head

    @property
    def is_constant(self):
        return self.expression.is_constant

    @property
    def is_syntactic(self):
        return self.expression.is_syntactic

    @property
    def variables(self):
        variables = self.expression.variables
        variables[self.name] += 1
        return variables

    @property
    def symbols(self):
        return self.expression.symbols

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

    @staticmethod
    def fixed(name: str, length: int, constraint:Optional[Constraint]=None):
        """Creates a `Variable` with :class:`Wildcard` that matches exactly `length` expressions."""
        return Variable(name, Wildcard.dot(length), constraint)

    def preorder_iter(self, predicate:Optional[ExpressionPredicate]=None, position:Tuple[int,...]=()) -> Iterator[Tuple['Expression',Tuple[int,...]]]:
        if predicate is None or predicate(self):
            yield self, position
        yield from self.expression.preorder_iter(predicate, position + (0, ))

    def _is_linear(self, variables):
        if self.name in variables:
            return False
        variables.add(self.name)
        return True

    def __str__(self):
        if isinstance(self.expression, Wildcard):
            value = self.name + str(self.expression)
        else:
            value = '%s_: %s' % (self.name, self.expression)
        if self.constraint:
            value += ' /; %s' % str(self.constraint)

        return value

    def __repr__(self):
        if self.constraint:
            return '%s(%r, %r, constraint=%r)' % (self.__class__.__name__, self.name, self.expression, self.constraint)
        return '%s(%r, %r)' % (self.__class__.__name__, self.name, self.expression)

    def __eq__(self, other):
        return isinstance(other, Variable) and self.name == other.name and self.expression == other.expression

    def __lt__(self, other):
        if isinstance(other, Symbol):
            return False
        if isinstance(other, Variable):
            return self.name < other.name
        return type(self).__name__ < type(other).__name__

    def __getitem__(self, key):
        if len(key) == 0:
            return self
        if key[0] != 0:
            raise IndexError('Invalid position.')
        return self.expression[key[1:]]

    def __hash__(self):
        return hash((type(self), self.name, self.expression))


class Wildcard(Atom):
    """A wildcard that matches any expression.

    The wildcard will match any number of expressions between `min_count` and `fixed_size`.
    Optionally, the wildcard can also be constrained to only match expressions satisfying a predicate.
    """

    def __init__(self, min_count: int, fixed_size: bool, constraint: Optional[Constraint]=None) -> None:
        """
        Arguments:
            min_count
                The minimum number of expressions this wildcard will match. Must be a non-negative number.
            fixed_size
                If `True`, the wildcard matches exactly `min_count` expressions.
                If `False`, the wildcard is a sequence wildcard and can match `min_count` or more expressions.
            constraint
                An optional constraint for expressions to be considered a match. If set, this
                callback is invoked for every match and the return value is utilized to decide
                whether the match is valid.
        """
        if min_count < 0:
            raise ValueError("min_count cannot be negative")
        if min_count == 0 and fixed_size:
            raise ValueError("Cannot create a fixed zero length wildcard")

        super().__init__(constraint)
        self.min_count = min_count
        self.fixed_size = fixed_size

    @property
    def is_constant(self):
        return False

    @property
    def is_syntactic(self):
        return self.fixed_size

    @staticmethod
    def dot(length:int=1):
        """Creates a :class:`Wildcard` that matches a fixed number `length` of arguments.
        
        Defaults to matching only a single argument."""
        return Wildcard(min_count=length, fixed_size=True)

    @staticmethod
    def star():
        """Creates a :class:`Wildcard` that matches any number of arguments."""
        return Wildcard(min_count=0, fixed_size=False)

    @staticmethod
    def plus():
        """Creates a :class:`Wildcard` that matches at least one and up to any number of arguments."""
        return Wildcard(min_count=1, fixed_size=False)

    def __str__(self):
        if not self.fixed_size:
            if self.min_count == 0:
                return '___'
            elif self.min_count == 1:
                return '__'
        return '_'

    def __repr__(self):
        if self.constraint:
            return '%s(%r, %r, constraint=%r)' % (self.__class__.__name__, self.min_count, self.fixed_size, self.constraint)
        return '%s(%r, %r)' % (self.__class__.__name__, self.min_count, self.fixed_size)

    def __lt__(self, other):
        return isinstance(other, Wildcard)

    def __eq__(self, other):
        return isinstance(other, Wildcard) and \
               other.min_count == self.min_count and \
               other.fixed_size == self.fixed_size

    def __hash__(self):
        return hash((type(self), self.min_count, self.fixed_size))

if __name__ == '__main__':
    f = Operation.new('f', Arity.binary, associative=True, commutative=True, one_identity=True)
    g = Operation.new('g', Arity.binary)
    x = Variable.dot('x')
    y = Variable.star('y')
    z = Variable.plus('z')
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')

    expr = f(f(b), a)

    print(repr(expr))
    print(repr(f))
