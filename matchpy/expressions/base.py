# -*- coding: utf-8 -*-
"""This module contains the `base class <Expression>` for all expressions.

See `expressions.expressions` for the actual basic expression building blocks.
"""
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Callable, Iterator, Optional, Set, Tuple, Dict, Union

from multiset import Multiset

if TYPE_CHECKING:
    from .constraints import Constraint  # pylint: disable=unused-import
else:
    Substitution = Dict[str, Union[Tuple['Expression', ...], Multiset['Expression'], 'Expression']]
    Constraint = Callable[[Substitution], bool]

__all__ = ['Expression']

ExprPredicate = Optional[Callable[['Expression'], bool]]
ExpressionsWithPos = Iterator[Tuple['Expression', Tuple[int, ...]]]


class Expression(metaclass=ABCMeta):
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

    __slots__ = 'constraint', 'head'

    def __init__(self, constraint: 'Constraint' =None) -> None:
        """Create a new expression.

        Args:
            constraint:
                An optional constraint expression, which is checked for each match
                to verify it.
        """
        self.constraint = constraint
        self.head = None  # type: Union[type, Atom]

    @property
    def variables(self) -> Multiset[str]:
        """A multiset of the variable names occurring in the expression."""
        return self._variables()

    @staticmethod
    def _variables() -> Multiset[str]:
        return Multiset()

    @property
    def symbols(self) -> Multiset[str]:
        """A multiset of the symbol names occurring in the expression."""
        return self._symbols()

    @staticmethod
    def _symbols() -> Multiset[str]:
        return Multiset()

    @property
    def is_constant(self) -> bool:
        """True, iff the expression does not contain any wildcards."""
        return self._is_constant()

    @staticmethod
    def _is_constant() -> bool:
        return True

    @property
    def is_syntactic(self) -> bool:
        """True, iff the expression does not contain any associative or commutative operations or sequence wildcards."""
        return self._is_syntactic()

    @staticmethod
    def _is_syntactic() -> bool:
        return True

    @property
    def is_linear(self) -> bool:
        """True, iff the expression is linear, i.e. every variable may occur at most once."""
        return self._is_linear(set())

    @staticmethod
    def _is_linear(variables: Set[str]) -> bool:  # pylint: disable=unused-argument
        return True

    @property
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
