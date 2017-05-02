# -*- coding: utf-8 -*-
"""This module contains the pattern building blocks."""
import math
# pylint: disable=unused-import
from typing import Optional, Type
# pylint: enable=unused-import

from multiset import Multiset

from .expressions import Atom, Expression, Symbol, Tuple, Callable, Iterator

__all__ = [
    'Wildcard', 'SymbolWildcard', 'Pattern', 'ExpressionSequence', 'Repeated', 'Alternatives', 'make_dot_variable',
    'make_plus_variable', 'make_star_variable', 'make_symbol_variable'
]

ExprPredicate = Optional[Callable[['Expression'], bool]]
ExpressionsWithPos = Iterator[Tuple['Expression', Tuple[int, ...]]]


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
    """

    head = None

    def __init__(self, min_count: int, fixed_size: bool, variable_name=None) -> None:
        """
        Args:
            min_count:
                The minimum number of expressions this wildcard will match. Must be a non-negative number.
            fixed_size:
                If ``True``, the wildcard matches exactly *min_count* expressions.
                If ``False``, the wildcard is a sequence wildcard and can match *min_count* or more expressions.

        Raises:
            ValueError: if *min_count* is negative or when trying to create a fixed zero-length wildcard.
        """
        if min_count < 0:
            raise ValueError("min_count cannot be negative")
        if min_count == 0 and fixed_size:
            raise ValueError("Cannot create a fixed zero length wildcard")

        super().__init__(variable_name)
        self.min_count = min_count
        self.fixed_size = fixed_size
        self.min_length = min_count
        self.max_length = min_count if fixed_size else math.inf

    def _is_constant(self) -> bool:
        return False

    def _is_syntactic(self) -> bool:
        return self.fixed_size

    def with_renamed_vars(self, renaming) -> 'Wildcard':
        return type(self)(
            self.min_count, self.fixed_size, variable_name=renaming.get(self.variable_name, self.variable_name)
        )

    @staticmethod
    def dot(name=None) -> 'Wildcard':
        """Create a `Wildcard` that matches a single argument.

        Args:
            name: An optional name for the wildcard.

        Returns:
            A dot wildcard.
        """
        return Wildcard(min_count=1, fixed_size=True, variable_name=name)

    @staticmethod
    def symbol(name: str=None, symbol_type: Type[Symbol]=Symbol) -> 'SymbolWildcard':
        """Create a `SymbolWildcard` that matches a single `Symbol` argument.

        Args:
            name:
                Optional variable name for the wildcard.
            symbol_type:
                An optional subclass of `Symbol` to further limit which kind of symbols are
                matched by the wildcard.

        Returns:
            A `SymbolWildcard` that matches the *symbol_type*.
        """
        if isinstance(name, type) and issubclass(name, Symbol) and symbol_type is Symbol:
            return SymbolWildcard(name)
        return SymbolWildcard(symbol_type, variable_name=name)

    @staticmethod
    def star(name=None) -> 'Wildcard':
        """Creates a `Wildcard` that matches any number of arguments.

        Args:
            name:
                Optional variable name for the wildcard.

        Returns:
            A star wildcard.
        """
        return Wildcard(min_count=0, fixed_size=False, variable_name=name)

    @staticmethod
    def plus(name=None) -> 'Wildcard':
        """Creates a `Wildcard` that matches at least one and up to any number of arguments

        Args:
            name:
                Optional variable name for the wildcard.

        Returns:
            A plus wildcard.
        """
        return Wildcard(min_count=1, fixed_size=False, variable_name=name)

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
        if self.variable_name:
            value = '{}{}'.format(self.variable_name, value)
        return value

    def __repr__(self):
        if self.variable_name:
            return '{!s}({!r}, {!r}, variable_name={})'.format(
                type(self).__name__, self.min_count, self.fixed_size, self.variable_name
            )
        return '{!s}({!r}, {!r})'.format(type(self).__name__, self.min_count, self.fixed_size)

    def __lt__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        if not isinstance(other, Wildcard):
            return type(self).__name__ < type(other).__name__
        if self.min_count != other.min_count or self.fixed_size != other.fixed_size:
            return self.min_count < other.min_count or (self.fixed_size and not other.fixed_size)
        if self.variable_name != other.variable_name:
            return (self.variable_name or '') < (other.variable_name or '')
        if not isinstance(self, SymbolWildcard):
            return isinstance(other, SymbolWildcard)
        if isinstance(other, SymbolWildcard):
            return self.symbol_type.__name__ < other.symbol_type.__name__
        return False

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            other.min_count == self.min_count and other.fixed_size == self.fixed_size and
            self.variable_name == other.variable_name
        )

    def __hash__(self):
        return hash((Wildcard, self.min_count, self.fixed_size, self.variable_name))

    def __copy__(self) -> 'Wildcard':
        return type(self)(self.min_count, self.fixed_size, variable_name=self.variable_name)


class SymbolWildcard(Wildcard):
    """A special `Wildcard` that matches a `Symbol`.

    Attributes:
        symbol_type:
            A subclass of `Symbol` to constrain what the wildcard matches.
            If not specified, the wildcard will match any `Symbol`.
    """

    def __init__(self, symbol_type: Type[Symbol]=Symbol, variable_name=None) -> None:
        """
        Args:
            symbol_type:
                A subclass of `Symbol` to constrain what the wildcard matches.
                If not specified, the wildcard will match any `Symbol`.

        Raises:
            TypeError: if *symbol_type* is not a subclass of `Symbol`.
        """
        super().__init__(1, True, variable_name)

        if not issubclass(symbol_type, Symbol):
            raise TypeError("The type constraint must be a subclass of Symbol")

        self.symbol_type = symbol_type

    def with_renamed_vars(self, renaming) -> 'SymbolWildcard':
        return type(self)(self.symbol_type, variable_name=renaming.get(self.variable_name, self.variable_name))

    def __eq__(self, other):
        return (
            isinstance(other, type(self)) and self.symbol_type == other.symbol_type and
            self.variable_name == other.variable_name
        )

    def __hash__(self):
        return hash((SymbolWildcard, self.symbol_type, self.variable_name))

    def __repr__(self):
        if self.variable_name:
            return '{!s}({!r}, variable_name={})'.format(type(self).__name__, self.symbol_type, self.variable_name)
        return '{!s}({!r})'.format(type(self).__name__, self.symbol_type)

    def __str__(self):
        if self.variable_name:
            return '{}_[{!s}]'.format(self.variable_name, self.symbol_type.__name__)
        return '_[{!s}]'.format(self.symbol_type.__name__)

    def __copy__(self) -> 'SymbolWildcard':
        return type(self)(self.symbol_type, self.variable_name)


class ExpressionSequence(Expression):
    def __init__(self, *children, variable_name=None):
        super().__init__(variable_name)
        self.children = children
        self.head = children[0].head if len(children) > 0 else None
        self.min_length = sum(c.min_length for c in children)
        self.max_length = sum(c.max_length for c in children)

    def __new__(cls, *children):
        if len(children) == 1:
            return children[0]
        return super(ExpressionSequence, cls).__new__(cls)

    def _is_constant(self) -> bool:
        return all(c.is_constant for c in self.children)

    def _is_syntactic(self) -> bool:
        return all(c.is_syntactic for c in self.children)

    def with_renamed_vars(self, renaming) -> 'Wildcard':
        return type(self)(*(c.with_renamed_vars(renaming) for c in self.children),
            variable_name=renaming.get(self.variable_name, self.variable_name))

    def collect_variables(self, variables) -> None:
        super().collect_variables(variables)
        for child in self.children:
            child.collect_variables(variables)

    def collect_symbols(self, symbols) -> None:
        super().collect_symbols(symbols)
        for child in self.children:
            child.collect_symbols(symbols)

    def _preorder_iter(self, predicate: ExprPredicate=None, position: Tuple[int, ...]=()) -> ExpressionsWithPos:
        if predicate is None or predicate(self):
            yield self, position
        for i, child in enumerate(self.children):
            yield from child._preorder_iter(predicate, position + (i, ))  # pylint: disable=protected-access

    def __eq__(self, other):
        return (isinstance(other, type(self)) and self.children == other.children and self.variable_name == other.variable_name)

    def __lt__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        if isinstance(other, Symbol):
            return False
        if not isinstance(other, type(self)):
            return type(other).__name__ < type(self).__name__
        if len(self.children) != len(other.children):
            return len(self.children) < len(other.children)
        for left, right in zip(self.children, other.children):
            if left < right:
                return True
            elif right < left:
                return False
        return (self.variable_name or '') < (other.variable_name or '')

    def __getitem__(self, key: Tuple[int, ...]) -> Expression:
        if len(key) == 0:
            return self
        if key[0] > len(self.children):
            raise IndexError("Invalid position.")
        return self.children[key[0]][key[1:]]

    def __str__(self):
        return '({!s})'.format(', '.join(str(c) for c in self.children))

    def __repr__(self):
        return '{!s}({!s})'.format(type(self).__name__, ', '.join(repr(c) for c in self.children))

    def __hash__(self):
        return hash((type(self), ) + tuple(self.children))

    def __copy__(self) -> 'ExpressionSequence':
        return type(self)(*self.children, variable_name=self.variable_name)


class Alternatives(Expression):
    def __init__(self, *children, variable_name=None):
        super().__init__(variable_name)
        self.children = sorted(children)
        self.head = None
        self.min_length = min(c.min_length for c in children)
        self.max_length = max(c.max_length for c in children)

    def __new__(cls, *children, variable_name=None):
        if len(children) < 1:
            raise ValueError("Must have at least one alternative")
        if len(children) == 1:
            return children[0]
        if any(isinstance(c, ExpressionSequence) and len(c.children) == 0 for c in children):
            new_children = [c for c in children if not isinstance(c, ExpressionSequence) or len(c.children) != 0]
            return Repeated(Alternatives(*new_children), 0, 1, variable_name)
        return super(Alternatives, cls).__new__(cls)

    def _is_constant(self) -> bool:
        return False

    def _is_syntactic(self) -> bool:
        return False

    def with_renamed_vars(self, renaming) -> 'Wildcard':
        return type(self)(*(c.with_renamed_vars(renaming) for c in self.children),
            variable_name=renaming.get(self.variable_name, self.variable_name))

    def collect_variables(self, variables) -> None:
        super().collect_variables(variables)
        common = self.children[0].variables
        for child in self.children[1:]:
            common.intersection_update(child.variables)
        variables.update(common)

    def collect_symbols(self, symbols) -> None:
        super().collect_symbols(symbols)
        common = self.children[0].symbols
        for child in self.children[1:]:
            common.intersection_update(child.symbols)
        symbols.update(common)

    def _preorder_iter(self, predicate: ExprPredicate=None, position: Tuple[int, ...]=()) -> ExpressionsWithPos:
        if predicate is None or predicate(self):
            yield self, position
        for i, child in enumerate(self.children):
            yield from child._preorder_iter(predicate, position + (i, ))  # pylint: disable=protected-access

    def __eq__(self, other):
        return (isinstance(other, type(self)) and self.children == other.children and self.variable_name == other.variable_name)

    def __lt__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        if isinstance(other, Symbol):
            return False
        if not isinstance(other, type(self)):
            return type(other).__name__ < type(self).__name__
        if len(self.children) != len(other.children):
            return len(self.children) < len(other.children)
        for left, right in zip(self.children, other.children):
            if left < right:
                return True
            elif right < left:
                return False
        return (self.variable_name or '') < (other.variable_name or '')

    def __getitem__(self, key: Tuple[int, ...]) -> Expression:
        if len(key) == 0:
            return self
        if key[0] > len(self.children):
            raise IndexError("Invalid position.")
        return self.children[key[0]][key[1:]]

    def __str__(self):
        return '({!s})'.format(' | '.join(str(c) for c in self.children))

    def __repr__(self):
        return '{!s}({!s})'.format(type(self).__name__, ', '.join(repr(c) for c in self.children))

    def __hash__(self):
        return hash((type(self), ) + tuple(self.children))

    def __copy__(self) -> 'Alternatives':
        return type(self)(*self.children, variable_name=self.variable_name)


class Repeated(Expression):
    def __init__(self, expression, min_count=1, max_count=math.inf, variable_name=None):
        if max_count < 1:
            raise ValueError('The max_count must be positive')
        if min_count < 0:
            raise ValueError('The min_count must not be negative')
        if max_count < min_count:
            raise ValueError('The max_count must not be smaller than the min_count')
        super().__init__(variable_name)
        self.expression = expression
        self.min_count = min_count
        self.max_count = max_count
        self.head = expression.head
        self.min_length = expression.min_length * min_count
        self.max_length = expression.max_length * max_count

    def __new__(cls, expression, min_count=1, max_count=math.inf, variable_name=None):
        if min_count == max_count:
            sequence = [expression] * min_count
            return ExpressionSequence(sequence)
        return super(Repeated, cls).__new__(cls)

    def _is_constant(self) -> bool:
        return False

    def _is_syntactic(self) -> bool:
        return False

    def with_renamed_vars(self, renaming) -> 'Repeated':
        return type(self)(
            self.expression.with_renamed_vars(renaming), self.min_count, self.max_count,
            renaming.get(self.variable_name, self.variable_name)
        )

    def collect_variables(self, variables) -> None:
        super().collect_variables(variables)
        self.expression.collect_variables(variables)

    def collect_symbols(self, symbols) -> None:
        super().collect_symbols(symbols)
        self.expression.collect_symbols(symbols)

    def _preorder_iter(self, predicate: ExprPredicate=None, position: Tuple[int, ...]=()) -> ExpressionsWithPos:
        if predicate is None or predicate(self):
            yield self, position
        yield from self.expression._preorder_iter(predicate, position + (0, ))  # pylint: disable=protected-access

    def __eq__(self, other):
        return (
            isinstance(other, type(self)) and self.expression == other.expression and
            self.min_count == other.min_count and self.max_count == other.max_count and
            self.variable_name == other.variable_name
        )

    def __lt__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        if isinstance(other, Symbol):
            return False
        if not isinstance(other, type(self)):
            return type(other).__name__ < type(self).__name__
        if self.expression != other.expression:
            return self.expression < other.expression
        if self.max_count != other.max_count:
            return self.max_count < other.max_count
        if self.min_count != other.min_count:
            return self.min_count < other.min_count
        return (self.variable_name or '') < (other.variable_name or '')

    def __getitem__(self, key: Tuple[int, ...]) -> Expression:
        if len(key) == 0:
            return self
        if key[0] != 0:
            raise IndexError("Invalid position.")
        return self.expression[key[1:]]

    def __str__(self):
        if self.max_count == math.inf:
            if self.min_count == 1:
                return '({})+'.format(self.expression)
            elif self.min_count == 0:
                return '({})*'.format(self.expression)
        elif self.min_count == 0 and self.max_count == 1:
            return '({})?'.format(self.expression)
        return '({}){{{}, {}}}'.format(self.expression, self.min_count, self.max_count)

    def __repr__(self):
        return '{}({}, min_count={}, max_count={})'.format(
            type(self).__name__, self.expression, self.min_count, self.max_count
        )

    def __hash__(self):
        return hash((type(self), self.expression))

    def __copy__(self) -> 'Repeated':
        return type(self)(self.expression, self.min_count, self.max_count, self.variable_name)


class Pattern:
    """A pattern is a term that can be matched against another subject term.

    A pattern can contain variables and can optionally have constraints attached to it.
    Those constraints a predicates which limit what the pattern can match.
    """

    def __init__(self, expression, *constraints) -> None:
        """
        Args:
            expression:
                The term that forms the pattern.
            *constraints:
                Optional constraints for the pattern.
        """
        self.expression = expression
        self.constraints = constraints

    def __str__(self):
        if not self.constraints:
            return str(self.expression)
        return '{} /; {}'.format(self.expression, ' and '.join(map(str, self.constraints)))

    def __repr__(self):
        if not self.constraints:
            return '{}({})'.format(type(self).__name__, self.expression)
        return '{}({}, constraints={})'.format(type(self).__name__, self.expression, self.constraints)

    def __eq__(self, other):
        if not isinstance(other, Pattern):
            return NotImplemented
        return self.expression == other.expression and self.constraints == other.constraints

    @property
    def is_syntactic(self):
        """True, iff the pattern is :term:`syntactic`."""
        return self.expression.is_syntactic

    @property
    def local_constraints(self):
        """The subset of the patterns contrainst which are local.

        A local constraint has a defined non-empty set of dependency variables.
        These constraints can be evaluated once their dependency variables have a substitution.
        """
        return [c for c in self.constraints if c.variables]

    @property
    def global_constraints(self):
        """The subset of the patterns contrainst which are global.

        A global constraint does not define dependency variables and can only be evaluated, once the
        match has been completed.
        """
        return [c for c in self.constraints if not c.variables]


def make_dot_variable(name):
    """Create a new variable with the given name that matches a single term.

    Args:
        name:
            The name of the variable

    Returns:
        The new dot variable.
    """
    return Wildcard.dot(name)


def make_symbol_variable(name, symbol_type=Symbol):
    """Create a new variable with the given name that matches a single symbol.

    Optionally, a symbol type can be specified to further limit what the variable can match.

    Args:
        name:
            The name of the variable
        symbol_type:
            The symbol type must be a subclass of `Symbol`. Defaults to `Symbol` itself.

    Returns:
        The new symbol variable.
    """
    return Wildcard.symbol(name, symbol_type)


def make_star_variable(name):
    """Create a new variable with the given name that matches any number of terms.

    Can also match an empty argument sequence.

    Args:
        name:
            The name of the variable

    Returns:
        The new star variable.
    """
    return Wildcard.star(name)


def make_plus_variable(name):
    """Create a new variable with the given name that matches any number of terms.

    Only matches sequences with at least one argument.

    Args:
        name:
            The name of the variable

    Returns:
        The new plus variable.
    """
    return Wildcard.plus(name)
