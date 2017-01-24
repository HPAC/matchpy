# -*- coding: utf-8 -*-
"""Contains functions to make expressions :term:`frozen`, so that they become :term:`hashable`.

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
from abc import ABCMeta
from typing import Iterable

from multiset import Multiset

from .expressions import Expression, _OperationMeta, Operation, Variable, Wildcard, Symbol, SymbolWildcard
from ..utils import cached_property

__all__ = ['FrozenExpression', 'freeze', 'unfreeze']

_module_name = '{}.'.format(__name__.split('.')[0])


class _FrozenMeta(ABCMeta):
    """Metaclass for :class:`FrozenExpression`."""
    __call__ = type.__call__


class _FrozenOperationMeta(_FrozenMeta, _OperationMeta, ABCMeta):
    """Metaclass that mixes :class:`_FrozenMeta` and :class:`_OperationMeta`.

    Used to override the :meth:`from_args` to create modified frozen operations easily.
    """

    def from_args(cls, *args, **kwargs):
        """Create a new :term:`frozen` instance of the class using the given arguments.

        Overrides :meth:`_OperationMeta.from_args`.
        This will first create a new instance of the unfrozen base operation and then freeze that using the
        :class:`FrozenExpression` initializer.

        Args:
            *args:
                Positional arguments for the operation initializer.
            **kwargs:
                Keyword arguments for the operation initializer.
        """
        for base in cls.__mro__:
            if isinstance(base, _OperationMeta) and not isinstance(base, _FrozenMeta):
                return FrozenExpression.__new__(cls, base(*args, **kwargs))
        assert False, "unreachable, unless an invalid frozen operation subclass was created manually"

    def __repr__(cls):
        return cls._repr('FrozenOperation')


class FrozenExpression(Expression, metaclass=_FrozenMeta):  # pylint: disable=abstract-method,too-many-instance-attributes
    """Base class for :term:`frozen` expressions.

    .. warning::

        DO NOT instantiate this class directly, use :func:`freeze` instead!

    Only use this class for :func:`isinstance` checks:

    >>> isinstance(a, FrozenExpression)
    False
    >>> isinstance(freeze(a), FrozenExpression)
    True
    """

    def __new__(cls, expr: Expression) -> 'FrozenExpression':
        if cls is FrozenExpression:
            raise TypeError('Cannot instantiate FrozenExpression class directly.')

        # This is a bit of a hack...
        # Copy all attributes of the original expression over to this instance
        # Use the _frozen attribute to lock changing attributes after the copying is done
        self = Expression.__new__(cls)

        object.__setattr__(self, '_frozen', False)
        expr.copy_to(self)

        if isinstance(expr, Operation):
            self.operands = tuple(freeze(e) for e in expr.operands)
            self.head = expr.head
        elif isinstance(expr, Symbol):
            self.head = self
        elif isinstance(expr, Variable):
            self.expression = freeze(expr.expression)
            self.head = self.expression.head

        object.__setattr__(self, '_frozen', True)

        return self

    def __init__(self, expr):  # pylint: disable=super-init-not-called
        # All the work is done in __new__()
        pass

    def __setattr__(self, name, value):
        if self._frozen:  # pylint: disable=no-member
            raise TypeError("Cannot modify a FrozenExpression")
        else:
            object.__setattr__(self, name, value)

    @cached_property
    def without_constraints(self) -> 'FrozenExpression':
        """Cached version of :attr:`.Expression.without_constraints`."""
        return freeze(super().without_constraints)

    def with_renamed_vars(self, renaming) -> 'FrozenExpression':
        """Frozen version of :meth:`.Expression.with_renamed_vars`."""
        return freeze(super().with_renamed_vars(renaming))


_frozen_type_cache = {}  # type: Dict[Type[Expression], Type[FrozenExpression]]
"""Caches the frozen types generated for expression types."""


def _frozen_new(cls, *args, **kwargs):
    """__new__ for FrozenExpression subclasses.

    Wraps the original base's constructor in a call to freeze().
    """
    return freeze(cls._original_base(*args, **kwargs))  # pylint: disable=protected-access


def freeze(expression: Expression) -> FrozenExpression:
    """Return a :term:`frozen` version of the expression.

    The new type for the frozen expression is created dynamically as a subclass of both :class:`FrozenExpression`
    and the type of the original expression. If the expression is already frozen, it is returned unchanged.

    Args:
        expression: The expression to freeze.

    Returns:
        The frozen expression.
    """
    # pylint: disable=protected-access
    if isinstance(expression, FrozenExpression):
        return expression
    base = type(expression)
    if base not in _frozen_type_cache:
        meta = _FrozenOperationMeta if isinstance(base, _OperationMeta) else _FrozenMeta
        frozen_class = meta(
            'Frozen' + base.__name__,
            (FrozenExpression, base),
            {
                '_original_base': base,
                '__new__': _frozen_new
            }
        )  # yapf: disable
        _frozen_type_cache[base] = frozen_class
    else:
        frozen_class = _frozen_type_cache[base]

    return FrozenExpression.__new__(frozen_class, expression)


def unfreeze(expression: FrozenExpression) -> Expression:
    """Return a non-:term:`frozen` version of the expression.

    This function reverts :func:`freeze`. A mutable version is created from the expression using its original class
    without the FrozenExpression mixin. If the given expression is not frozen, it is returned unchanged.

    Args:
        expression: The expression to unfreeze.

    Returns:
        The unfrozen expression.
    """
    if not isinstance(expression, FrozenExpression):
        return expression
    if isinstance(expression, SymbolWildcard):
        return SymbolWildcard(expression.symbol_type, expression.constraint)
    if isinstance(expression, Wildcard):
        return Wildcard(expression.min_count, expression.fixed_size, expression.constraint)
    if isinstance(expression, Variable):
        return Variable(expression.name, unfreeze(expression.expression), expression.constraint)
    if isinstance(expression, Symbol):
        return expression._original_base(expression.name)  # pylint: disable=protected-access
    if isinstance(expression, Operation):
        return expression._original_base.from_args(  # pylint: disable=protected-access
            *map(unfreeze, expression.operands), constraint=expression.constraint
        )
    assert False, "Unreachable, unless new types of expressions are added that are unsupported"
