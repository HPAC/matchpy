# -*- coding: utf-8 -*-
"""This module contains the expression classes."""

"""TODO work on this (outdated)

Contains functions to make expressions :term:`frozen`, so that they become :term:`hashable`.

Normal expressions are mutable and hence not :term:`hashable`:

>>> expr = f(b, x_)
>>> print(expr)
f(b, x_)
>>> expr.operands = [a, x_]
>>> print(expr)
f(a, x_)
>>> hash(expr)
Traceback (most recent call last):
...
TypeError: unhashable type: 'f'

Use the `freeze()` function to freeze an expression and make it :term:`hashable`:

>>> frozen = freeze(expr)
>>> frozen == expr
True
>>> print(frozen)
f(a, x_)
>>> hash(frozen) == hash(frozen)
True

Attempting to modify a :term:`frozen` `.Expression` will raise an exception:

>>> frozen.operands = [a]
Traceback (most recent call last):
...
TypeError: Cannot modify a FrozenExpression

You can check whether an expression is :term:`frozen` using `isinstance`:

>>> isinstance(frozen, FrozenExpression)
True
>>> isinstance(expr, FrozenExpression)
False

Expressions can be unfrozen using `unfreeze()`:

>>> unfrozen = unfreeze(frozen)
>>> unfrozen == expr
True
>>> print(unfrozen)
f(a, x_)

This expression can be modified again:

>>> unfrozen.operands = [a]
>>> print(unfrozen)
f(a)

This does not affect the original expression or the :term:`frozen` one:

>>> print(frozen)
f(a, x_)
>>> print(expr)
f(a, x_)
"""
import itertools
import keyword
from abc import ABCMeta, abstractmethod
from enum import Enum, EnumMeta
from typing import (Callable, Iterator, List,  # pylint: disable=unused-import
                    NamedTuple, Optional, Set, Tuple, TupleMeta, Type, Union)

from multiset import Multiset

from . import constraints  # pylint: disable=unused-import
from ..utils import cached_property

__all__ = ['MutableExpression', 'freeze', 'unfreeze', 'FrozenExpression', 'Expression', 'Arity', 'Atom', 'Symbol', 'Variable', 'Wildcard', 'Operation', 'SymbolWildcard']

ExprPredicate = Optional[Callable[['Expression'], bool]]
ExpressionsWithPos = Iterator[Tuple['Expression', Tuple[int, ...]]]

class ExpressionMeta(ABCMeta):
    def __init__(cls, name, bases, dct):
        ABCMeta.__init__(cls, name, bases, dct)
        if cls.__module__ == __name__ and cls.__name__ in ('Expression', 'MutableExpression', 'FrozenExpression'):
            return
        if not issubclass(cls, (MutableExpression, FrozenExpression)):
            cls.generic_base_type = cls


class Expression(metaclass=ExpressionMeta):
    """Base class for all expressions.

    Do not subclass this class directly but rather :class:`Symbol` or :class:`Operation`.
    Creating a direct subclass of Expression might break several (matching) algorithms.

    Attributes:
        constraint (Constraint):
            An optional constraint expression, which is checked for each match
            to verify it.
        head (Optional[Union[type, Atom]]):
            The head of the expression. For an operation, it is the type of the operation (i.e. a subclass of
            :class:`Operation`). For wildcards, it is ``None``. For symbols, it is the symbol itself. For a variable,
            it is the variable's inner :attr:`~Variable.expression`.
    """

    def __init__(self, constraint: 'constraints.Constraint' =None) -> None:
        """Create a new expression.

        Args:
            constraint:
                An optional constraint expression, which is checked for each match
                to verify it.
        """
        self.constraint = constraint

    def __new__(cls, *args, **kwargs):
        if not issubclass(cls, (FrozenExpression, MutableExpression)):
            cls = _create_mixin_type(cls, MutableExpression, _mutable_type_cache)
            return cls.__new__(cls, *args, **kwargs)
        return object.__new__(cls)

    @cached_property
    def variables(self) -> Multiset[str]:
        """A multiset of the variable names occurring in the expression."""
        return self._variables()

    @staticmethod
    def _variables() -> Multiset[str]:
        return Multiset()

    @cached_property
    def symbols(self) -> Multiset[str]:
        """A multiset of the symbol names occurring in the expression."""
        return self._symbols()

    @staticmethod
    def _symbols() -> Multiset[str]:
        return Multiset()

    @cached_property
    def is_constant(self) -> bool:
        """True, iff the expression does not contain any wildcards."""
        return self._is_constant()

    @staticmethod
    def _is_constant() -> bool:
        return True

    @cached_property
    def is_syntactic(self) -> bool:
        """True, iff the expression does not contain any associative or commutative operations or sequence wildcards."""
        return self._is_syntactic()

    @staticmethod
    def _is_syntactic() -> bool:
        return True

    @cached_property
    def is_linear(self) -> bool:
        """True, iff the expression is linear, i.e. every variable may occur at most once."""
        return self._is_linear(set())

    @staticmethod
    def _is_linear(variables: Set[str]) -> bool:  # pylint: disable=unused-argument
        return True

    @cached_property
    def without_constraints(self) -> 'Expression':
        """A copy of the expression without constraints."""
        return self._without_constraints()

    @abstractmethod
    def _without_constraints(self) -> 'Expression':
        raise NotImplementedError()

    @abstractmethod
    def with_renamed_vars(self, renaming) -> 'Expression':
        """Return a copy of the expression with renamed variables."""
        raise NotImplementedError()

    def copy_to(self, other: 'Expression') -> None:
        """Copy the expressions attributes to the other one."""
        other.constraint = self.constraint

    def preorder_iter(self, predicate: ExprPredicate=None) -> ExpressionsWithPos:
        """Iterates over all subexpressions that match the (optional) `predicate`.

        Args:
            predicate:
                A predicate to filter what expressions are yielded. It gets the expression and if it returns ``True``,
                the expression is yielded.

        Yields:
            Every subexpression along with a position tuple. Each item in the tuple is the position of an operation
            operand:

                - ``()`` is the position of the root element
                - ``(0, )`` that of its first operand
                - ``(0, 1)`` the position of the second operand of the root's first operand.
                - etc.

            A variable's expression always has the position ``0`` relative to the variable, i.e. if the root is a
            variable, then its expression has the position ``(0, )``.
        """
        yield from self._preorder_iter(predicate, ())

    def _preorder_iter(self, predicate: ExprPredicate, position: Tuple[int, ...]) -> ExpressionsWithPos:
        if predicate is None or predicate(self):
            yield self, position

    def __getitem__(self, position: Tuple[int, ...]) -> 'Expression':
        """Return the subexpression at the given position.

        Args:
            position: The position as a tuple. See :meth:`preorder_iter` for its format.

        Returns:
            The subexpression at the given position.

        Raises:
            IndexError: If the position is invalid, i.e. it refers to a non-existing subexpression.
        """
        if len(position) == 0:
            return self
        raise IndexError("Invalid position")

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError()


class FrozenExpression(Expression):
    prefix = 'Frozen'


class MutableExpression(Expression):  # pylint: disable=abstract-method,too-many-instance-attributes
    """Base class for :term:`frozen` expressions.

    .. warning::

        DO NOT instantiate this class directly, use :func:`freeze` instead!

    Only use this class for :func:`isinstance` checks:

    >>> isinstance(a, FrozenExpression)
    False
    >>> isinstance(freeze(a), FrozenExpression)
    True
    """

    prefix = 'Mutable'

    __hash__ = None


# This class is needed so that Tuple and Enum play nicely with each other
class _ArityMeta(TupleMeta, EnumMeta):
    @classmethod
    def __prepare__(mcs, cls, bases, **kwargs):  # pylint: disable=unused-argument
        return super().__prepare__(cls, bases)


_ArityBase = NamedTuple('_ArityBase', [('min_count', int), ('fixed_size', bool)])


class Arity(_ArityBase, Enum, metaclass=_ArityMeta):
    """Arity of an operator as (`int`, `bool`) tuple.

    The first component is the minimum number of operands.
    If the second component is ``True``, the operator has fixed width arity. In that case, the first component
    describes the fixed number of operands required.
    If it is ``False``, the operator has variable width arity.
    """

    def __new__(cls, *data):
        return tuple.__new__(cls, data)

    nullary = (0, True)
    unary = (1, True)
    binary = (2, True)
    ternary = (3, True)
    polyadic = (2, False)
    variadic = (0, False)

    def __repr__(self):
        return "{!s}.{!s}".format(type(self).__name__, self._name_)


class _OperationMeta(ExpressionMeta):
    """Metaclass for `Operation`

    This metaclass is mainly used to override :meth:`__call__` to provide simplification when creating a
    new operation expression. This is done to avoid problems when overriding ``__new__`` of the operation class
    and clashes with the `.FrozenOperation` class.
    """

    def __init__(cls, name, bases, dct):
        super(_OperationMeta, cls).__init__(name, bases, dct)

        if cls.arity[1] and cls.one_identity:
            raise TypeError('{}: An operation with fixed arity cannot have one_identity = True.'.format(name))

        if cls.arity == Arity.unary and cls.infix:
            raise TypeError('{}: Unary operations cannot use infix notation.'.format(name))

        if not issubclass(cls, (MutableExpression, FrozenExpression)):
            cls.head = cls

    def __repr__(cls):
        if cls is Operation:
            return super().__repr__()
        flags = []
        if cls.associative:
            flags.append('associative')
        if cls.commutative:
            flags.append('commutative')
        if cls.one_identity:
            flags.append('one_identity')
        if cls.infix:
            flags.append('infix')
        return '{}[{!r}, {!r}, {}]'.format(cls.__name__, cls.name, cls.arity, ', '.join(flags))

    def __str__(cls):
        return cls.name

    def __call__(cls, *operands: Expression, constraint: 'constraints.Constraint' =None):
        # __call__ is overridden, so that for one_identity operations with a single argument
        # that argument can be returned instead
        operands = list(operands)
        if not cls._simplify(operands):
            return operands[0]

        operation = Expression.__new__(cls)
        operation.__init__(*operands, constraint=constraint)

        return operation

    def _simplify(cls, operands: List[Expression]) -> bool:
        """Flatten/sort the operands of associative/commutative operations.

        Returns:
            False iff *one_identity* is True and the operation contains a single
            argument that is not a sequence wildcard.
        """

        if cls.associative:
            new_operands = []  # type: List[Expression]
            for operand in operands:
                if isinstance(operand, cls):
                    new_operands.extend(operand.operands)  # type: ignore
                else:
                    new_operands.append(operand)
            operands.clear()
            operands.extend(new_operands)

        if cls.one_identity and len(operands) == 1:
            expr = operands[0]
            if isinstance(expr, Variable):
                expr = expr.expression
            if not isinstance(expr, Wildcard) or (expr.min_count == 1 and expr.fixed_size):
                return False

        if cls.commutative:
            operands.sort()

        return True

    def from_args(cls, *args, **kwargs):
        """Create a new instance of the class using the given arguments.

        This is used so that it can be overridden by `._FrozenOperationMeta`. Use this to create a new instance
        of an operation with changed operands instead of using the instantiation operation directly.
        Because the  `.FrozenExpression` has a different initializer, this would break for :term:`frozen`
        operations otherwise.

        Args:
            *args:
                Positional arguments for the operation initializer.
            **kwargs:
                Keyword arguments for the operation initializer.
        """
        return cls(*args, **kwargs)


class Operation(Expression, metaclass=_OperationMeta):
    """Base class for all operations.

    Do not instantiate this class directly, but create a subclass for every operation in your domain.
    You can use :meth:`new` as a shortcut for doing so.
    """

    name = None  # type: str
    """str: Name or symbol for the operator.

    This needs to be overridden in the subclass.
    """

    arity = Arity.variadic  # type: Arity
    """Arity: The arity of the operator.

    Trying to construct an operation expression with a number of operands that does not fit its
    operation's arity will result in an error.
    """

    associative = False
    """bool: True if the operation is associative, i.e. `f(a, f(b, c)) = f(f(a, b), c)`.

    This attribute is used to flatten nested associative operations of the same type.
    Therefore, the `arity` of an associative operation has to have an unconstrained maximum
    number of operand.
    """

    commutative = False
    """bool: True if the operation is commutative, i.e. `f(a, b) = f(b, a)`.

    Note that commutative operations will always be converted into canonical
    form with sorted operands.
    """

    one_identity = False
    """bool: True if the operation with a single argument is equivalent to the identity function.

    This property is used to simplify expressions, e.g. for ``f`` with ``f.one_identity = True``
    the expression ``f(a)`` if simplified to ``a``.
    """

    infix = False
    """bool: True if the name of the operation should be used as an infix operator by str()."""

    def __init__(self, *operands: Expression, constraint: 'constraints.Constraint' =None) -> None:
        """Create an operation expression.

        Args:
            *operands
                The operands for the operation expression.
            constraint
                An optional constraint expression, which is checked for each match
                to verify it.

        Raises:
            ValueError:
                if the operand count does not match the operation's arity.
            ValueError:
                if the operation contains conflicting variables, i.e. variables with the same name that match
                different things. A common example would be mixing sequence and fixed variables with the same name in
                one expression.
        """
        super().__init__(constraint)

        operand_count, variable_count = self._count_operands(operands)

        if not variable_count and operand_count < self.arity.min_count:
            raise ValueError(
                "Operation {!s} got arity {!s}, but got {:d} operands.".
                format(type(self).__name__, self.arity, len(operands))
            )

        if self.arity.fixed_size and operand_count > self.arity.min_count:
            msg = "Operation {!s} got arity {!s}, but got {:d} operands.".format(
                type(self).__name__, self.arity, operand_count
            )
            if self.associative:
                msg += " Associative operations should have a variadic/polyadic arity."
            raise ValueError(msg)

        variables = dict()
        var_iters = [o.preorder_iter(lambda e: isinstance(e, Variable)) for o in operands]
        for var, _ in itertools.chain.from_iterable(var_iters):
            if var.name in variables:
                if variables[var.name] != var.without_constraints:
                    raise ValueError(
                        "Conflicting versions of variable {!s}: {!r} vs {!r}".
                        format(var.name, var, variables[var.name])
                    )
            else:
                variables[var.name] = var.without_constraints

        self.operands = list(operands)

    @staticmethod
    def _count_operands(operands):
        operand_count = 0
        for operand in operands:
            if isinstance(operand, Variable):
                operand = operand.expression
            if isinstance(operand, Wildcard):
                if operand.fixed_size:
                    operand_count += operand.min_count
                else:
                    return 0, True
            else:
                operand_count += 1
        return operand_count, False

    def __str__(self):
        if self.infix:
            separator = ' {!s} '.format(self.name) if self.name else ''
            value = '({!s})'.format(separator.join(str(o) for o in self.operands))
        else:
            value = '{!s}({!s})'.format(self.name, ', '.join(str(o) for o in self.operands))
        if self.constraint:
            value += ' /; {!s}'.format(self.constraint)
        return value

    def __repr__(self):
        operand_str = ', '.join(map(repr, self.operands))
        if self.constraint:
            return '{!s}({!s}, constraint={!r})'.format(type(self).__name__, operand_str, self.constraint)
        return '{!s}({!s})'.format(type(self).__name__, operand_str)

    @staticmethod
    def new(
            name: str,
            arity: Arity,
            class_name: str=None,
            *,
            associative: bool=False,
            commutative: bool=False,
            one_identity: bool=False,
            infix: bool=False
    ) -> Type['Operation']:
        """Utility method to create a new operation type.

        Example:

        >>> Times = Operation.new('*', Arity.polyadic, 'Times', associative=True, commutative=True, one_identity=True)
        >>> Times
        Times['*', Arity.polyadic, associative, commutative, one_identity]
        >>> str(Times(Symbol('a'), Symbol('b')))
        '*(a, b)'

        Args:
            name:
                Name or symbol for the operator. Will be used as name for the new class if
                `class_name` is not specified.
            arity:
                The arity of the operator as explained in the documentation of `Operation`.
            class_name:
                Name for the new operation class to be used instead of name. This argument
                is required if `name` is not a valid python identifier.

        Keyword Args:
            associative:
                See :attr:`~Operation.associative`.
            commutative:
                See :attr:`~Operation.commutative`.
            one_identity:
                See :attr:`~Operation.one_identity`.
            infix:
                See :attr:`~Operation.infix`.

        Raises:
            ValueError: if the class name of the operation is not a valid class identifier.
        """
        class_name = class_name or name
        if not class_name.isidentifier() or keyword.iskeyword(class_name):
            raise ValueError("Invalid identifier for new operator class.")

        return type(
            class_name, (Operation, ), {
                'name': name,
                'arity': arity,
                'associative': associative,
                'commutative': commutative,
                'one_identity': one_identity,
                'infix': infix
            }
        )

    def __lt__(self, other):
        if isinstance(other, Symbol):
            return False

        if not isinstance(other, self.generic_base_type):
            return self.generic_base_type.__name__ < other.generic_base_type.__name__

        if len(self.operands) != len(other.operands):
            return len(self.operands) < len(other.operands)

        for left, right in zip(self.operands, other.operands):
            if left < right:
                return True
            elif right < left:
                return False

        return False

    def __eq__(self, other):
        if not isinstance(other, self.generic_base_type):
            return NotImplemented
        return (
            self.constraint == other.constraint and len(self.operands) == len(other.operands) and
            all(x == y for x, y in zip(self.operands, other.operands))
        )

    def __getitem__(self, key: Tuple[int, ...]) -> Expression:
        if len(key) == 0:
            return self
        head, *remainder = key
        return self.operands[head][remainder]

    __getitem__.__doc__ = Expression.__getitem__.__doc__

    def _is_constant(self) -> bool:
        return all(x.is_constant for x in self.operands)

    def _is_syntactic(self) -> bool:
        if self.associative or self.commutative:
            return False
        return all(o.is_syntactic for o in self.operands)

    def _variables(self) -> Multiset[str]:
        return sum((x.variables for x in self.operands), Multiset())

    def _symbols(self) -> Multiset[str]:
        return sum((x.symbols for x in self.operands), Multiset([self.name]))

    def _without_constraints(self):
        return type(self).from_args(*(o.without_constraints for o in self.operands))

    def _is_linear(self, variables: Set[str]) -> bool:
        return all(o._is_linear(variables) for o in self.operands)  # pylint: disable=protected-access

    def _preorder_iter(self, predicate: ExprPredicate=None, position: Tuple[int, ...]=()) -> ExpressionsWithPos:
        if predicate is None or predicate(self):
            yield self, position
        for i, operand in enumerate(self.operands):
            yield from operand._preorder_iter(predicate, position + (i, ))  # pylint: disable=protected-access

    def __hash__(self):
        return hash((self.name, self.constraint) + tuple(self.operands))

    def with_renamed_vars(self, renaming) -> 'Operation':
        constraint = self.constraint.with_renamed_vars(renaming) if self.constraint else None
        return type(self).from_args(*(o.with_renamed_vars(renaming) for o in self.operands), constraint=constraint)

    def copy_to(self, other: 'Operation') -> None:
        super().copy_to(other)
        other.operands = self.operands[:]


class Atom(Expression):  # pylint: disable=abstract-method
    """Base for all atomic expressions."""
    pass


class Symbol(Atom):
    """An atomic constant expression term.

    It is uniquely identified by its name.

    Attributes:
        name (str):
            The symbol's name.
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name:
                The name of the symbol that uniquely identifies it.
        """
        super().__init__(None)
        self.name = name
        self.head = self

    def __str__(self):
        return self.name

    def __repr__(self):
        if self.constraint:
            return '{!s}({!r}, constraint={!r})'.format(type(self).__name__, self.name, self.constraint)
        return '{!s}({!r})'.format(type(self).__name__, self.name)

    def _symbols(self):
        return Multiset([self.name])

    def _without_constraints(self):
        return type(self)(self.name)

    def with_renamed_vars(self, renaming) -> 'Symbol':
        return type(self)(self.name)

    def copy_to(self, other: 'Symbol') -> None:
        super().copy_to(other)
        other.head = other
        other.name = self.name

    def __lt__(self, other):
        if isinstance(other, Symbol):
            return self.name < other.name
        return True

    def __eq__(self, other):
        if not isinstance(other, self.generic_base_type):
            return NotImplemented
        return self.name == other.name and self.constraint == other.constraint

    def __hash__(self):
        return hash((Symbol, self.name, self.constraint))


class Variable(Expression):
    """A variable that is captured during a match.

    Wraps another pattern expression that is used to match. On match, the matched
    value is captured for the variable.

    Attributes:
        name (str):
            The name of the variable that is used to capture its value in the match.
            Can be used to access its value in constraints or for replacement.
        expression (Expression):
            The expression that is used for matching. On match, its value will be
            assigned to the variable. Usually, a `Wildcard` is used to match
            any expression.
        constraint (Optional[Constraint]):
            See :attr:`Expression.constraint`.

    """

    def __init__(self, name: str, expression: Expression, constraint: 'constraints.Constraint' =None) -> None:
        """
        Args:
            name:
                The name of the variable that is used to capture its value in the match.
                Can be used to access its value in constraints or for replacement.
            expression:
                The expression that is used for matching. On match, its value will be
                assigned to the variable. Usually, a `Wildcard` is used to match
                any expression.
            constraint:
                See `.Expression`.

        Raises:
            ValueError: if the expression contains a variable. Nested variables are not supported.
        """
        super().__init__(constraint)

        if expression.variables:
            raise ValueError("Cannot have nested variables in expression.")

        if expression.is_constant:
            raise ValueError("Cannot have constant expression for a variable.")

        self.name = name
        self.expression = expression
        self.head = expression.head

    def _is_constant(self) -> bool:
        return self.expression.is_constant

    def _is_syntactic(self) -> bool:
        return self.expression.is_syntactic

    def _variables(self) -> Multiset[str]:
        return Multiset([self.name])

    def _symbols(self) -> Multiset[str]:
        return self.expression.symbols

    def _without_constraints(self):
        return type(self)(self.name, self.expression.without_constraints)

    def with_renamed_vars(self, renaming) -> 'Variable':
        constraint = self.constraint.with_renamed_vars(renaming) if self.constraint else None
        name = renaming.get(self.name, self.name)
        return type(self)(name, self.expression.with_renamed_vars(renaming), constraint)

    @staticmethod
    def dot(name: str, constraint: 'constraints.Constraint' =None) -> 'Variable':
        """Create a `Variable` with a `Wildcard` that matches exactly one argument.

        Args:
            name:
                The name of the variable.
            constraint:
                An optional `.Constraint` which can filter what is matched by the variable.

        Returns:
            A `Variable` with a `Wildcard` that matches exactly one argument.
        """
        return Variable(name, Wildcard.dot(), constraint)

    @staticmethod
    def symbol(name: str, symbol_type: Type[Symbol]=Symbol, constraint: 'constraints.Constraint' =None) -> 'Variable':
        """Create a `Variable` with a `SymbolWildcard`.

        Args:
            name:
                The name of the variable.
            symbol_type:
                An optional subclass of `Symbol` to further limit which kind of symbols are
                matched by the wildcard.
            constraint:
                An optional `.Constraint` which can filter what is matched by the variable.

        Returns:
            A `Variable` that matches a `Symbol` with type *symbol_type*
        """
        return Variable(name, Wildcard.symbol(symbol_type), constraint)

    @staticmethod
    def star(name: str, constraint: 'constraints.Constraint' =None) -> 'Variable':
        """Creates a `Variable` with `Wildcard` that matches any number of arguments.

        Args:
            name:
                The name of the variable.
            constraint:
                An optional `.Constraint` which can filter what is matched by the variable.

        Returns:
            A `Variable` with a `Wildcard` that matches any number of arguments.
        """
        return Variable(name, Wildcard.star(), constraint)

    @staticmethod
    def plus(name: str, constraint: 'constraints.Constraint' =None) -> 'Variable':
        """Creates a `Variable` with `Wildcard` that matches at least one and up to any number of arguments.

        Args:
            name:
                The name of the variable.
            constraint:
                An optional `.Constraint` which can filter what is matched by the variable.

        Returns:
            A `Variable` with a `Wildcard` that matches at least one and up to any number of arguments.
        """
        return Variable(name, Wildcard.plus(), constraint)

    @staticmethod
    def fixed(name: str, length: int, constraint: 'constraints.Constraint' =None) -> 'Variable':
        """Creates a `Variable` with `Wildcard` that matches exactly *length* expressions.

        Args:
            name:
                The name of the variable.
            length:
                The length of the variable.
            constraint:
                An optional `.Constraint` which can filter what is matched by the variable.

        Returns:
            A `Variable` with `Wildcard` that matches exactly *length* expressions.
        """
        return Variable(name, Wildcard.dot(length), constraint)

    def _preorder_iter(self, predicate: ExprPredicate=None, position: Tuple[int, ...]=()) -> ExpressionsWithPos:
        if predicate is None or predicate(self):
            yield self, position
        yield from self.expression._preorder_iter(predicate, position + (0, ))  # pylint: disable=protected-access

    def _is_linear(self, variables: Set[str]) -> bool:
        if self.name in variables:
            return False
        variables.add(self.name)
        return True

    def __str__(self):
        if isinstance(self.expression, Wildcard):
            value = self.name + str(self.expression)
        else:
            value = '{!s}: {!s}'.format(self.name, self.expression)
        if self.constraint:
            value += ' /; {!s}'.format(str(self.constraint))

        return value

    def __repr__(self):
        if self.constraint:
            return '{!s}({!r}, {!r}, constraint={!r})'.format(
                type(self).__name__, self.name, self.expression, self.constraint
            )
        return '{!s}({!r}, {!r})'.format(type(self).__name__, self.name, self.expression)

    def __eq__(self, other):
        return (
            isinstance(other, self.generic_base_type) and self.name == other.name and self.expression == other.expression and
            self.constraint == other.constraint
        )

    def __lt__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        if isinstance(other, Symbol):
            return False
        if isinstance(other, Variable):
            return self.name < other.name
        return self.generic_base_type.__name__ < other.generic_base_type.__name__

    def __getitem__(self, key: Tuple[int, ...]) -> Expression:
        if len(key) == 0:
            return self
        if key[0] != 0:
            raise IndexError("Invalid position.")
        return self.expression[key[1:]]

    def __hash__(self):
        return hash((Variable, self.name, self.expression, self.constraint))

    def copy_to(self, other: 'Variable') -> None:
        super().copy_to(other)
        other.name = self.name
        other.expression = self.expression


class Wildcard(Atom):
    """A wildcard that matches any expression.

    The wildcard will match any number of expressions between *min_count* and *fixed_size*.
    Optionally, the wildcard can also be constrained to only match expressions satisfying a predicate.

    Attributes:
        min_count (int):
            The minimum number of expressions this wildcard will match.
        fixed_size (bool):
            If ``True``, the wildcard matches exactly *min_count* expressions.
            If ``False``, the wildcard is a sequence wildcard and can match *min_count* or more expressions.
        constraint (Optional[Constraint]):
            An optional constraint for expressions to be considered a match. If set, this
            callback is invoked for every match and the return value is utilized to decide
            whether the match is valid.
    """

    head = None

    def __init__(self, min_count: int, fixed_size: bool, constraint: 'constraints.Constraint' =None) -> None:
        """
        Args:
            min_count:
                The minimum number of expressions this wildcard will match. Must be a non-negative number.
            fixed_size:
                If ``True``, the wildcard matches exactly *min_count* expressions.
                If ``False``, the wildcard is a sequence wildcard and can match *min_count* or more expressions.
            constraint:
                An optional constraint for expressions to be considered a match. If set, this
                callback is invoked for every match and the return value is utilized to decide
                whether the match is valid.

        Raises:
            ValueError: if *min_count* is negative or when trying to create a fixed zero-length wildcard.
        """
        if min_count < 0:
            raise ValueError("min_count cannot be negative")
        if min_count == 0 and fixed_size:
            raise ValueError("Cannot create a fixed zero length wildcard")

        super().__init__(constraint)
        self.min_count = min_count
        self.fixed_size = fixed_size

    def _is_constant(self) -> bool:
        return False

    def _is_syntactic(self) -> bool:
        return self.fixed_size

    def _without_constraints(self):
        return type(self)(self.min_count, self.fixed_size)

    def with_renamed_vars(self, renaming) -> 'Wildcard':
        constraint = self.constraint.with_renamed_vars(renaming) if self.constraint else None
        return type(self)(self.min_count, self.fixed_size, constraint)

    @staticmethod
    def dot(length: int=1) -> 'Wildcard':
        """Create a `Wildcard` that matches a fixed number *length* of arguments.

        Defaults to matching only a single argument.

        Args:
            length: The fixed number of arguments to match.

        Returns:
            A wildcard with a fixed size.
        """
        return Wildcard(min_count=length, fixed_size=True)

    @staticmethod
    def symbol(symbol_type: Type[Symbol]=Symbol) -> 'SymbolWildcard':
        """Create a `SymbolWildcard` that matches a single `Symbol` argument.

        Args:
            symbol_type:
                An optional subclass of `Symbol` to further limit which kind of smybols are
                matched by the wildcard.

        Returns:
            A `SymbolWildcard` that matches the *symbol_type*.
        """
        return SymbolWildcard(symbol_type)

    @staticmethod
    def star() -> 'Wildcard':
        """Creates a `Wildcard` that matches any number of arguments.

        Returns:
            A wildcard that matches any number of arguments.
        """
        return Wildcard(min_count=0, fixed_size=False)

    @staticmethod
    def plus() -> 'Wildcard':
        """Creates a `Wildcard` that matches at least one and up to any number of arguments

        Returns:
            A wildcard that matches at least one and up to any number of arguments
        """
        return Wildcard(min_count=1, fixed_size=False)

    def __str__(self):
        value = None
        if not self.fixed_size:
            if self.min_count == 0:
                value = '___'
            elif self.min_count == 1:
                value = '__'
        elif self.min_count == 1:
            value = '_'
        if value is None:
            value = '_[{:d}{!s}]'.format(self.min_count, '' if self.fixed_size else '+')
        if self.constraint:
            value += ' /; {!s}'.format(str(self.constraint))
        return value

    def __repr__(self):
        if self.constraint:
            return '{!s}({!r}, {!r}, constraint={!r})'.format(
                type(self).__name__, self.min_count, self.fixed_size, self.constraint
            )
        return '{!s}({!r}, {!r})'.format(type(self).__name__, self.min_count, self.fixed_size)

    def __lt__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        if not isinstance(other, Wildcard):
            return self.generic_base_type.__name__ < other.generic_base_type.__name__
        return self.min_count < other.min_count or (self.fixed_size and not other.fixed_size)

    def __eq__(self, other):
        if not isinstance(other, self.generic_base_type):
            return NotImplemented
        return (
            other.min_count == self.min_count and other.fixed_size == self.fixed_size and
            other.constraint == self.constraint
        )

    def __hash__(self):
        return hash((Wildcard, self.min_count, self.fixed_size, self.constraint))

    def copy_to(self, other: 'Wildcard') -> None:
        super().copy_to(other)
        other.min_count = self.min_count
        other.fixed_size = self.fixed_size


class SymbolWildcard(Wildcard):
    """A special `Wildcard` that matches a `Symbol`.

    Attributes:
        symbol_type:
            A subclass of `Symbol` to constraint what the wildcard matches.
            If not specified, the wildcard will match any `Symbol`.
    """

    def __init__(self, symbol_type: Type[Symbol]=Symbol, constraint: 'constraints.Constraint' =None) -> None:
        """
        Args:
            symbol_type:
                A subclass of `Symbol` to constraint what the wildcard matches.
                If not specified, the wildcard will match any `Symbol`.
            constraint:
                An optional constraint for expressions to be considered a match. If set, this
                callback is invoked for every match and the return value is utilized to decide
                whether the match is valid.

        Raises:
            TypeError: if *symbol_type* is not a subclass of `Symbol`.
        """
        super().__init__(1, True, constraint)

        if not issubclass(symbol_type, Symbol):
            raise TypeError("The type constraint must be a subclass of Symbol")

        self.symbol_type = symbol_type

    def _without_constraints(self):
        return type(self)(self.symbol_type)

    def with_renamed_vars(self, renaming) -> 'SymbolWildcard':
        constraint = self.constraint.with_renamed_vars(renaming) if self.constraint else None
        return type(self)(self.symbol_type, constraint)

    def __eq__(self, other):
        return (
            isinstance(other, self.generic_base_type) and self.symbol_type == other.symbol_type and
            other.constraint == self.constraint
        )

    def __hash__(self):
        return hash((SymbolWildcard, self.symbol_type, self.constraint))

    def __repr__(self):
        if self.constraint:
            return '{!s}({!r}, constraint={!r})'.format(type(self).__name__, self.symbol_type, self.constraint)
        return '{!s}({!r})'.format(type(self).__name__, self.symbol_type)

    def __str__(self):
        if self.constraint:
            return '_[{!s}] /; {!s}'.format(self.symbol_type.__name__, self.constraint)
        return '_[{!s}]'.format(self.symbol_type.__name__)

    def copy_to(self, other: 'SymbolWildcard') -> None:
        super().copy_to(other)
        other.symbol_type = self.symbol_type


_frozen_type_cache = {
    Expression: FrozenExpression,
    MutableExpression: FrozenExpression
}  # type: Dict[Type[Expression], Type[FrozenExpression]]
_mutable_type_cache = {
    Expression: MutableExpression,
    FrozenExpression: MutableExpression
}  # type: Dict[Type[Expression], Type[MutableExpression]]


def _create_mixin_type(cls, mixin, cache):
    if issubclass(cls, (FrozenExpression, MutableExpression)):
        cls = next(
            b for b in cls.__mro__
            if issubclass(b, Expression) and not issubclass(b, (FrozenExpression, MutableExpression))
        )
    if cls not in cache:
        name = cls.__name__
        bases = [cls]
        for base in cls.__bases__:
            if issubclass(base, Expression):
                bases.append(_create_mixin_type(base, mixin, cache))
        if all(not issubclass(b, mixin) for b in bases):
            bases.append(mixin)
        cache[cls] = type(mixin.prefix + name, tuple(bases), {})
    return cache[cls]


def unfreeze(expression: Expression) -> MutableExpression:
    """Return a :term:`mutable` version of the expression.

    The new type for the mutable expression is created dynamically as a subclass of both :class:`MutableExpression`
    and the type of the original expression. If the expression is already mutable, it is returned unchanged.

    Args:
        expression: The expression to freeze.

    Returns:
        The frozen expression.
    """
    if isinstance(expression, MutableExpression):
        return expression
    if not isinstance(expression, FrozenExpression):
        raise TypeError("freeze: Expected a FrozenExpression, got {} instead.".format(type(expression).__name__))
    new_type = _create_mixin_type(type(expression), MutableExpression, _mutable_type_cache)
    new_expr = Expression.__new__(new_type)
    expression.copy_to(new_expr)

    if isinstance(expression, Operation):
        new_expr.operands = [unfreeze(e) for e in expression.operands]
    elif isinstance(expression, Symbol):
        new_expr.head = new_expr
    elif isinstance(expression, Variable):
        new_expr.expression = unfreeze(expression.expression)
        new_expr.head = new_expr.expression.head

    return new_expr


def freeze(expression: Expression) -> FrozenExpression:
    """Return a non-:term:`frozen` version of the expression.

    This function reverts :func:`freeze`. A mutable version is created from the expression using its original class
    without the FrozenExpression mixin. If the given expression is not frozen, it is returned unchanged.

    Args:
        expression: The expression to unfreeze.

    Returns:
        The unfrozen expression.
    """
    if isinstance(expression, FrozenExpression):
        return expression
    if not isinstance(expression, MutableExpression):
        raise TypeError("freeze: Expected a MutableExpression, got {} instead.".format(type(expression).__name__))
    new_type = _create_mixin_type(type(expression), FrozenExpression, _frozen_type_cache)
    new_expr = Expression.__new__(new_type)
    expression.copy_to(new_expr)

    if isinstance(expression, Operation):
        new_expr.operands = tuple(freeze(e) for e in expression.operands)
    elif isinstance(expression, Symbol):
        new_expr.head = new_expr
    elif isinstance(expression, Variable):
        new_expr.expression = freeze(expression.expression)
        new_expr.head = new_expr.expression.head

    return new_expr