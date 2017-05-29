# -*- coding: utf-8 -*-
"""This module contains the expression classes.

`Expressions <Expression>` can be used to model any kind of tree-like data structure. They consist of `operations
<Operation>` and `symbols <Symbol>`. In addition, `patterns <Pattern>` can be constructed, which may additionally,
contain `wildcards <Wildcard>` and variables.

You can define your own symbols and operations like this:

>>> f = Operation.new('f', Arity.variadic)
>>> a = Symbol('a')
>>> b = Symbol('b')

Then you can compose expressions out of these:

>>> print(f(a, b))
f(a, b)

For more information on how to create you own `operations <Operation>` and `symbols <Symbol>` you can look at their
documentation.

Normal expressions are immutable and hence :term:`hashable`:

>>> expr = f(b, x_)
>>> print(expr)
f(b, x_)
>>> hash(expr) == hash(expr)
True

Hence, some of the expression's properties are cached and nor updated when you modify them:

>>> expr.is_constant
False
>>> expr.operands = [a]
>>> expr.is_constant
False
>>> print(expr)
f(a)
>>> f(a).is_constant
True

Therefore, you should modify an expression but rather create a new one:

>>> expr2 = type(expr)(*[a])
>>> expr2.is_constant
True
>>> print(expr2)
f(a)
"""
from abc import ABCMeta
import keyword
from enum import Enum, EnumMeta
# pylint: disable=unused-import
from typing import (Callable, Iterator, List, NamedTuple, Optional, Set, Tuple, TupleMeta, Type, Union)
# pylint: enable=unused-import

from multiset import Multiset

from ..utils import cached_property

__all__ = [
    'Expression', 'Arity', 'Atom', 'Symbol', 'Wildcard', 'Operation', 'SymbolWildcard', 'Pattern', 'make_dot_variable',
    'make_plus_variable', 'make_star_variable', 'make_symbol_variable', 'AssociativeOperation', 'CommutativeOperation'
]

ExprPredicate = Optional[Callable[['Expression'], bool]]
ExpressionsWithPos = Iterator[Tuple['Expression', Tuple[int, ...]]]

MultisetOfStr = Multiset
MultisetOfVariables = Multiset


class Expression:
    """Base class for all expressions.

    Do not subclass this class directly but rather :class:`Symbol` or :class:`Operation`.
    Creating a direct subclass of Expression might break several (matching) algorithms.

    Attributes:
        head (Optional[Union[type, Atom]]):
            The head of the expression. For an operation, it is the type of the operation (i.e. a subclass of
            :class:`Operation`). For wildcards, it is ``None``. For symbols, it is the symbol itself.
    """

    def __init__(self, variable_name):
        self.variable_name = variable_name

    @cached_property
    def variables(self) -> MultisetOfVariables:
        """A multiset of the variables occurring in the expression."""
        variables = Multiset()
        self.collect_variables(variables)
        return variables

    def collect_variables(self, variables: MultisetOfVariables) -> None:
        """Recursively adds all variables occuring in the expression to the given multiset.

        This is used internally by `variables`. Needs to be overwritten by inheriting container expression classes.
        This method can be used when gathering the `variables` of multiple expressions, because only one multiset
        needs to be created and that is more efficient.

        Args:
            variables:
                Multiset of variables. All variables contained in the expression are recursively added to this multiset.
        """
        if self.variable_name is not None:
            variables.add(self.variable_name)

    @cached_property
    def symbols(self) -> MultisetOfStr:
        """A multiset of the symbol names occurring in the expression."""
        symbols = Multiset()
        self.collect_symbols(symbols)
        return symbols

    def collect_symbols(self, symbols: MultisetOfStr) -> None:
        """Recursively adds all symbols occuring in the expression to the given multiset.

        This is used internally by `symbols`. Needs to be overwritten by inheriting expression classes that
        can contain symbols. This method can be used when gathering the `symbols` of multiple expressions, because only
        one multiset needs to be created and that is more efficient.

        Args:
            symbols:
                Multiset of symbols. All symbols contained in the expression are recursively added to this multiset.
        """
        pass

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

    def __getitem__(self, position: Union[Tuple[int, ...], slice]) -> 'Expression':
        """Return the subexpression at the given position(s).

        It is also possible to use a slice notation to extract a sequence of subexpressions:

        >>> expr = f(a, b, a, c)
        >>> expr[(1, ):(2, )]
        [Symbol('b'), Symbol('a')]

        Args:
            position:
                The position as a tuple. See :meth:`preorder_iter` for its format.
                Alternatively, a range of positions can be passed using the slice notation.

        Returns:
            The subexpression at the given position(s).

        Raises:
            IndexError: If the position is invalid, i.e. it refers to a non-existing subexpression.
        """
        if isinstance(position, slice):
            if len(position.start) != len(position.stop):
                raise IndexError('Invalid slice: Start and stop must have the same length')
            if len(position.start) == 0:
                return [self]
            raise IndexError('Invalid slice: Parent expression is not an operation')
        if len(position) == 0:
            return self
        raise IndexError("Invalid position")

    def __contains__(self, expression: 'Expression') -> bool:
        return self == expression

    def __hash__(self):
        raise NotImplementedError()


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


class _OperationMeta(ABCMeta):
    """Metaclass for `Operation`

    This metaclass is mainly used to override :meth:`__call__` to provide simplification when creating a
    new operation expression. This is done to avoid problems when overriding ``__new__`` of the operation class.
    """

    def __init__(cls, name, bases, dct):
        super(_OperationMeta, cls).__init__(name, bases, dct)

        if cls.arity[1] and cls.one_identity:
            raise TypeError('{}: An operation with fixed arity cannot have one_identity = True.'.format(name))

        if cls.arity == Arity.unary and cls.infix:
            raise TypeError('{}: Unary operations cannot use infix notation.'.format(name))

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

    def __call__(cls, *operands: Expression, variable_name=None):
        # __call__ is overridden, so that for one_identity operations with a single argument
        # that argument can be returned instead
        operands = list(operands)
        one_identity_applies = cls._simplify(operands)
        if one_identity_applies:
            return operands[0]

        operation = Expression.__new__(cls)
        operation.__init__(operands, variable_name=variable_name)

        return operation

    def _simplify(cls, operands: List[Expression]) -> bool:
        """Flatten/sort the operands of associative/commutative operations.

        Returns:
            True iff *one_identity* is True and the operation contains a single
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
            if not isinstance(expr, Wildcard) or (expr.min_count == 1 and expr.fixed_size):
                return True

        if cls.commutative:
            operands.sort()

        return False


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

    def __init__(self, operands: List[Expression], variable_name=None) -> None:
        """Create an operation expression.

        Args:
            *operands
                The operands for the operation expression.

        Raises:
            ValueError:
                if the operand count does not match the operation's arity.
            ValueError:
                if the operation contains conflicting variables, i.e. variables with the same name that match
                different things. A common example would be mixing sequence and fixed variables with the same name in
                one expression.
        """
        super().__init__(variable_name)

        operand_count, variable_count = self._count_operands(operands)

        if not variable_count and operand_count < self.arity.min_count:
            raise ValueError(
                "Operation {!s} got arity {!s}, but got {:d} operands.".
                format(type(self).__name__, self.arity, operand_count)
            )

        if self.arity.fixed_size and operand_count > self.arity.min_count:
            msg = "Operation {!s} got arity {!s}, but got {:d} operands.".format(
                type(self).__name__, self.arity, operand_count
            )
            if self.associative:
                msg += " Associative operations should have a variadic/polyadic arity."
            raise ValueError(msg)

        self.operands = operands

    @staticmethod
    def _count_operands(operands):
        operand_count = 0
        variable = False
        for operand in operands:
            if isinstance(operand, Wildcard):
                operand_count += operand.min_count
                if not operand.fixed_size:
                    variable = True
            else:
                operand_count += 1
        return operand_count, variable

    def __str__(self):
        if self.infix:
            separator = ' {!s} '.format(self.name) if self.name else ''
            value = '({!s})'.format(separator.join(str(o) for o in self.operands))
        else:
            value = '{!s}({!s})'.format(self.name, ', '.join(str(o) for o in self.operands))
        if self.variable_name:
            value = '{}: {}'.format(self.variable_name, value)
        return value

    def __repr__(self):
        operand_str = ', '.join(map(repr, self.operands))
        if self.variable_name:
            return '{!s}({!s}, variable_name={})'.format(type(self).__name__, operand_str, self.variable_name)
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
        if not isinstance(other, Expression):
            return NotImplemented
        if not isinstance(other, type(self)) and not isinstance(self, type(other)):
            return type(self).__name__ < type(other).__name__
        if self.name != other.name:
            return self.name < other.name
        if len(self.operands) != len(other.operands):
            return len(self.operands) < len(other.operands)
        for left, right in zip(self.operands, other.operands):
            if left < right:
                return True
            elif right < left:
                return False
        return (self.variable_name or '') < (other.variable_name or '')

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            len(self.operands) == len(other.operands) and all(x == y for x, y in zip(self.operands, other.operands)) and
            self.variable_name == other.variable_name
        )

    def __iter__(self):
        return iter(self.operands)

    def __len__(self):
        return len(self.operands)

    def __getitem__(self, key: Union[Tuple[int, ...], slice]) -> Expression:
        if isinstance(key, int):
            return self.operands[key]
        if isinstance(key, slice):
            if len(key.start) != len(key.stop):
                raise IndexError('Invalid slice: Start and stop must have the same length')
            if len(key.start) == 0:
                return [self]
            if key.start > key.stop:
                raise IndexError('Invalid slice: Start must come before stop')
            if len(key.start) == 1:
                return self.operands[key.start[0]:key.stop[0] + 1]
            start, *new_start = key.start
            stop, *new_stop = key.stop
            if start != stop:
                raise IndexError('Invalid slice: Start and stop must have the same parent')
            return self.operands[start][new_start:new_stop]
        if isinstance(key, (list, tuple)):
            if len(key) == 0:
                return self
            head, *remainder = key
            return self.operands[head][remainder]
        raise TypeError('Invalid key: {}'.format(key))

    __getitem__.__doc__ = Expression.__getitem__.__doc__

    def __contains__(self, expression: 'Expression') -> bool:
        if self == expression:
            return True
        for operand in self.operands:
            if operand == expression:
                return True
            try:
                if expression in operand:
                    return True
            except TypeError:
                pass
        return False

    def _is_constant(self) -> bool:
        return all(x.is_constant for x in self.operands)

    def _is_syntactic(self) -> bool:
        if self.associative or self.commutative:
            return False
        return all(o.is_syntactic for o in self.operands)

    def collect_variables(self, variables) -> None:
        if self.variable_name:
            variables.add(self.variable_name)
        for operand in self.operands:
            operand.collect_variables(variables)

    def collect_symbols(self, symbols) -> None:
        symbols.add(self.name)
        for operand in self.operands:
            operand.collect_symbols(symbols)

    def _preorder_iter(self, predicate: ExprPredicate=None, position: Tuple[int, ...]=()) -> ExpressionsWithPos:
        if predicate is None or predicate(self):
            yield self, position
        for i, operand in enumerate(self.operands):
            yield from operand._preorder_iter(predicate, position + (i, ))  # pylint: disable=protected-access

    def __hash__(self):
        return hash((self.name, ) + tuple(self.operands))

    def with_renamed_vars(self, renaming) -> 'Operation':
        return type(self)(
            *(o.with_renamed_vars(renaming) for o in self.operands),
            variable_name=renaming.get(self.variable_name, self.variable_name)
        )

    def __copy__(self) -> 'Operation':
        return type(self)(*self.operands, variable_name=self.variable_name)


Operation.register(list)
Operation.register(tuple)
Operation.register(set)
Operation.register(frozenset)


class AssociativeOperation(metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, C):
        if cls is AssociativeOperation:
            if issubclass(C, Operation) and hasattr(C, 'associative'):
                return C.associative
        return NotImplemented


class CommutativeOperation(metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, C):
        if cls is CommutativeOperation:
            if issubclass(C, Operation) and hasattr(C, 'commutative'):
                return C.commutative
        return NotImplemented


CommutativeOperation.register(set)
CommutativeOperation.register(frozenset)


class Atom(Expression):  # pylint: disable=abstract-method
    """Base for all atomic expressions."""

    __iter__ = None


class Symbol(Atom):
    """An atomic constant expression term.

    It is uniquely identified by its name.

    Attributes:
        name (str):
            The symbol's name.
    """

    def __init__(self, name: str, variable_name=None) -> None:
        """
        Args:
            name:
                The name of the symbol that uniquely identifies it.
        """
        super().__init__(variable_name)
        self.name = name
        self.head = self

    def __str__(self):
        if self.variable_name:
            return '{}: {}'.format(self.name, self.variable_name)
        return self.name

    def __repr__(self):
        if self.variable_name:
            return '{!s}({!r}, variable_name={})'.format(type(self).__name__, self.name, self.variable_name)
        return '{!s}({!r})'.format(type(self).__name__, self.name)

    def collect_symbols(self, symbols):
        symbols.add(self.name)

    def with_renamed_vars(self, renaming) -> 'Symbol':
        return type(self)(self.name, variable_name=renaming.get(self.variable_name, self.variable_name))

    def __copy__(self) -> 'Symbol':
        return type(self)(self.name, variable_name=self.variable_name)

    def __lt__(self, other):
        if not isinstance(other, Expression):
            return NotImplemented
        if isinstance(other, Symbol):
            if self.name == other.name:
                return (self.variable_name or '') < (other.variable_name or '')
            return self.name < other.name
        return type(self).__name__ < type(other).__name__

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.name == other.name and self.variable_name == other.variable_name

    def __hash__(self):
        return hash((Symbol, self.name, self.variable_name))


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
