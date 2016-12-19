# -*- coding: utf-8 -*-
import itertools
import keyword
from abc import ABCMeta, abstractmethod
from enum import Enum, EnumMeta
from typing import (
    Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Set, Tuple, TupleMeta, Type, Union, cast,
    TYPE_CHECKING
)

from multiset import Multiset

from .utils import slot_cached_property

if TYPE_CHECKING:
    from .constraints import Constraint
else:
    Constraint = Callable[['Substitution'], bool]

__all__ = [
    'Arity', 'Expression', 'Atom', 'Symbol', 'Variable', 'Wildcard', 'Operation', 'SymbolWildcard', 'FrozenExpression',
    'freeze', 'unfreeze', 'Substitution'
]


# This class is needed so that Tuple and Enum play nicely with each other
class _ArityMeta(TupleMeta, EnumMeta):
    @classmethod
    def __prepare__(mcs, cls, bases, **kwargs):
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

    nullary = (0, True)
    unary = (1, True)
    binary = (2, True)
    ternary = (3, True)
    polyadic = (2, False)
    variadic = (0, False)

    def __repr__(self):
        return "{!s}.{!s}".format(type(self).__name__, self._name_)


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

    def __init__(self, constraint: Constraint=None) -> None:
        """Create a new expression.

        Args:
            constraint (Constraint):
                An optional constraint expression, which is checked for each match
                to verify it.
        """
        self.constraint = constraint
        self.head = None  # type: Union[type, Atom]

    @property
    def variables(self) -> Multiset[str]:
        """A multiset of the variable names occurring in the expression."""
        return self._variables()

    def _variables(self) -> Multiset[str]:
        return Multiset()

    @property
    def symbols(self) -> Multiset[str]:
        """A multiset of the symbol names occurring in the expression."""
        return self._symbols()

    def _symbols(self) -> Multiset[str]:
        return Multiset()

    @property
    def is_constant(self) -> bool:
        """True, iff the expression does not contain any wildcards."""
        return self._is_constant()

    def _is_constant(self) -> bool:
        return True

    @property
    def is_syntactic(self) -> bool:
        """True, iff the expression does not contain any associative or commutative operations or sequence wildcards."""
        return self._is_syntactic()

    def _is_syntactic(self) -> bool:
        return True

    @property
    def is_linear(self) -> bool:
        """True, iff the expression is linear, i.e. every variable may occur at most once."""
        return self._is_linear(set())

    def _is_linear(self, variables: Set[str]) -> bool:
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


class _OperationMeta(ABCMeta):
    """Metaclass for :class:`Operation`

    This metaclass is mainly used to override :meth:`__call__` to provide simplification when creating a
    new operation expression. This is done to avoid problems when overriding ``__new__`` of the operation class
    and clashes with the :class:`FrozenOperation` class.
    """

    def _repr(cls, name):  # pragma: no cover
        flags = []
        if cls.associative:
            flags.append('associative')
        if cls.commutative:
            flags.append('commutative')
        if cls.one_identity:
            flags.append('one_identity')
        if cls.infix:
            flags.append('infix')
        return '{}[{!r}, {!r}, {}]'.format(name, cls.name, cls.arity, ', '.join(flags))

    def __repr__(cls):
        return cls._repr('Operation')

    def __str__(cls):
        return cls.name

    def __call__(cls, *operands: Expression, constraint: Constraint=None):
        # __call__ is overridden, so that for one_identity operations with a single argument
        # that argument can be returned instead
        operands = list(operands)
        if not cls._simplify(operands):
            return operands[0]

        operation = object.__new__(cls)
        operation.__init__(*operands, constraint=constraint)

        return operation

    def _simplify(cls, operands: List[Expression]) -> bool:
        """Flatten/sort the operands of associative/commutative operations.

        Returns:
            False iff ``one_identity`` is True and the operation contains a single
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

        This is used so that it can be overridden by :class:`_FrozenOperationMeta`. Use this to create a new instance
        of an operation with changed operands instead of using the instantiation operation directly.
        Because the  :class:`FrozenExpression` has a different initializer, this would break for :term:`frozen`
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

    __slots__ = 'operands',

    def __init__(self, *operands: Expression, constraint: Constraint=None) -> None:
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
        self.head = type(self)

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
            value = '({!s})'.format((' {!s} '.format(self.name)).join(str(o) for o in self.operands))
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
        Operation['*', Arity.polyadic, associative, commutative, one_identity]
        >>> str(Times(Symbol('a'), Symbol('b')))
        '*(a, b)'

        Args:
            name:
                Name or symbol for the operator. Will be used as name for the new class if
                `class_name` is not specified.
            arity:
                The arity of the operator as explained in the documentation of :class:`Operation`.
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
                'infix': infix,
                '__slots__': ()
            }
        )

    def __lt__(self, other):
        if isinstance(other, Symbol):
            return False

        if not isinstance(other, type(self)):
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
        if not isinstance(self, type(other)):
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

    def _compute_hash(self):
        return hash((self.name, self.constraint) + tuple(self.operands))

    def with_renamed_vars(self, renaming) -> 'Operation':
        constraint = self.constraint.with_renamed_vars(renaming) if self.constraint else None
        return type(self).from_args(*(o.with_renamed_vars(renaming) for o in self.operands), constraint=constraint)


class Atom(Expression):  # pylint: disable=abstract-method
    pass


class Symbol(Atom):
    __slots__ = 'name',

    def __init__(self, name: str) -> None:
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
        if hasattr(self, '_original_base'):
            return freeze(self._original_base(self.name))
        return type(self)(self.name)

    def with_renamed_vars(self, renaming) -> 'Symbol':
        if hasattr(self, '_original_base'):
            return freeze(self._original_base(self.name))
        return type(self)(self.name)

    def __lt__(self, other):
        if isinstance(other, Symbol):
            return self.name < other.name
        return True

    def __eq__(self, other):
        if not isinstance(self, type(other)):
            return NotImplemented
        return self.name == other.name and self.constraint == other.constraint

    def _compute_hash(self):
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
            assigned to the variable. Usually, a :class:`Wildcard` is used to match
            any expression.
        constraint (Optional[Constraint]):
            See :attr:`Expression.constraint`.

    """

    __slots__ = 'name', 'expression'

    def __init__(self, name: str, expression: Expression, constraint: Constraint=None) -> None:
        """
        Args:
            name
                The name of the variable that is used to capture its value in the match.
                Can be used to access its value in constraints or for replacement.
            expression
                The expression that is used for matching. On match, its value will be
                assigned to the variable. Usually, a :class:`Wildcard` is used to match
                any expression.
            constraint
                See :class:`Expression`.

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
        return Variable(self.name, self.expression.without_constraints)

    def with_renamed_vars(self, renaming) -> 'Variable':
        constraint = self.constraint.with_renamed_vars(renaming) if self.constraint else None
        name = renaming.get(self.name, self.name)
        return Variable(name, self.expression.with_renamed_vars(renaming), constraint)

    @staticmethod
    def dot(name: str, constraint: Constraint=None) -> 'Variable':
        """Create a :class:`Variable` with a :class:`Wildcard` that matches exactly one argument."""
        return Variable(name, Wildcard.dot(), constraint)

    @staticmethod
    def symbol(name: str, symbol_type: Type[Symbol]=Symbol, constraint: Constraint=None) -> 'Variable':
        """Create a :class:`Variable` with a :class:`SymbolWildcard`.

        Args:
            name:
                The name of the variable.
            symbol_type:
                An optional subclass of :class:`Symbol` to further limit which kind of smybols are
                matched by the wildcard.
            constraint:
                An optional :class:`.Constraint` which can filter what is matched by the variable.

        Returns:
            A :class:`Variable` that matches a :class:`Symbol` with type ``symbol_type``.
        """
        return Variable(name, Wildcard.symbol(symbol_type), constraint)

    @staticmethod
    def star(name: str, constraint: Constraint=None) -> 'Variable':
        """Creates a `Variable` with :class:`Wildcard` that matches any number of arguments."""
        return Variable(name, Wildcard.star(), constraint)

    @staticmethod
    def plus(name: str, constraint: Constraint=None) -> 'Variable':
        """Creates a `Variable` with :class:`Wildcard` that matches at least one and up to any number of arguments."""
        return Variable(name, Wildcard.plus(), constraint)

    @staticmethod
    def fixed(name: str, length: int, constraint: Constraint=None) -> 'Variable':
        """Creates a `Variable` with :class:`Wildcard` that matches exactly `length` expressions."""
        return Variable(name, Wildcard.dot(length), constraint)

    def _preorder_iter(self, predicate: ExprPredicate=None, position: Tuple[int, ...]=()) -> ExpressionsWithPos:
        if predicate is None or predicate(self):
            yield self, position
        yield from self.expression._preorder_iter(predicate, position + (0, ))

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
            isinstance(other, Variable) and self.name == other.name and self.expression == other.expression and
            self.constraint == other.constraint
        )

    def __lt__(self, other):
        if isinstance(other, Symbol):
            return False
        if isinstance(other, Variable):
            return self.name < other.name
        return type(self).__name__ < type(other).__name__

    def __getitem__(self, key: Tuple[int, ...]) -> Expression:
        if len(key) == 0:
            return self
        if key[0] != 0:
            raise IndexError("Invalid position.")
        return self.expression[key[1:]]

    def _compute_hash(self):
        return hash((Variable, self.name, self.expression, self.constraint))


class Wildcard(Atom):
    """A wildcard that matches any expression.

    The wildcard will match any number of expressions between `min_count` and `fixed_size`.
    Optionally, the wildcard can also be constrained to only match expressions satisfying a predicate.

    Attributes:
        min_count (int):
            The minimum number of expressions this wildcard will match.
        fixed_size (bool):
            If `True`, the wildcard matches exactly `min_count` expressions.
            If `False`, the wildcard is a sequence wildcard and can match `min_count` or more expressions.
        constraint (Optional[Constraint]):
            An optional constraint for expressions to be considered a match. If set, this
            callback is invoked for every match and the return value is utilized to decide
            whether the match is valid.
    """

    __slots__ = 'min_count', 'fixed_size'

    def __init__(self, min_count: int, fixed_size: bool, constraint: Constraint=None) -> None:
        """
        Args:
            min_count:
                The minimum number of expressions this wildcard will match. Must be a non-negative number.
            fixed_size:
                If `True`, the wildcard matches exactly `min_count` expressions.
                If `False`, the wildcard is a sequence wildcard and can match `min_count` or more expressions.
            constraint:
                An optional constraint for expressions to be considered a match. If set, this
                callback is invoked for every match and the return value is utilized to decide
                whether the match is valid.

        Raises:
            ValueError: if ``min_count`` is negative or when trying to create a fixed zero-length wildcard.
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
        return Wildcard(self.min_count, self.fixed_size)

    def with_renamed_vars(self, renaming) -> 'Wildcard':
        constraint = self.constraint.with_renamed_vars(renaming) if self.constraint else None
        return Wildcard(self.min_count, self.fixed_size, constraint)

    @staticmethod
    def dot(length: int=1) -> 'Wildcard':
        """Create a :class:`Wildcard` that matches a fixed number `length` of arguments.

        Defaults to matching only a single argument.

        Args:
            length: The fixed number of arguments to match.

        Returns:
            A wildcard with a fixed size."""
        return Wildcard(min_count=length, fixed_size=True)

    @staticmethod
    def symbol(symbol_type: Type[Symbol]=Symbol) -> 'SymbolWildcard':
        """Create a :class:`SymbolWildcard` that matches a single :class:`Symbol` argument.

        Args:
            symbol_type:
                An optional subclass of :class:`Symbol` to further limit which kind of smybols are
                matched by the wildcard.

        Returns:
            A :class:`SymbolWildcard` that matches the ``symbol_type``.
        """
        return SymbolWildcard(symbol_type)

    @staticmethod
    def star() -> 'Wildcard':
        """Creates a :class:`Wildcard` that matches any number of arguments."""
        return Wildcard(min_count=0, fixed_size=False)

    @staticmethod
    def plus() -> 'Wildcard':
        """Creates a :class:`Wildcard` that matches at least one and up to any number of arguments."""
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
        return (not isinstance(other, Wildcard)) and type(self).__name__ < type(other).__name__

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            other.min_count == self.min_count and other.fixed_size == self.fixed_size and
            other.constraint == self.constraint
        )

    def _compute_hash(self):
        return hash((Wildcard, self.min_count, self.fixed_size, self.constraint))


class SymbolWildcard(Wildcard):
    """A special :class:`Wildcard` that matches a :class:`Symbol`."""

    __slots__ = 'symbol_type',

    def __init__(self, symbol_type: Type[Symbol]=Symbol, constraint: Constraint=None) -> None:
        """
        Args:
            symbol_type
                An optional subclass of :class:`Symbol` to further constraint what the wildcard matches.
                It will then only match symbols of that type.
            constraint
                An optional constraint for expressions to be considered a match. If set, this
                callback is invoked for every match and the return value is utilized to decide
                whether the match is valid.

        Raises:
            TypeError: if ``symbol_type`` is not a subclass of :class:`Symbol`.
        """
        super().__init__(1, True, constraint)

        if not issubclass(symbol_type, Symbol):
            raise TypeError("The type constraint must be a subclass of Symbol")

        self.symbol_type = symbol_type

    def _without_constraints(self):
        return SymbolWildcard(self.symbol_type)

    def with_renamed_vars(self, renaming) -> 'SymbolWildcard':
        constraint = self.constraint.with_renamed_vars(renaming) if self.constraint else None
        return SymbolWildcard(self.symbol_type, constraint)

    def __eq__(self, other):
        return (
            isinstance(other, SymbolWildcard) and self.symbol_type == other.symbol_type and
            other.constraint == self.constraint
        )

    def _compute_hash(self):
        return hash((SymbolWildcard, self.symbol_type, self.constraint))

    def __repr__(self):
        if self.constraint:
            return '{!s}({!r}, constraint={!r})'.format(type(self).__name__, self.symbol_type, self.constraint)
        return '{!s}({!r})'.format(type(self).__name__, self.symbol_type)

    def __str__(self):
        if self.constraint:
            return '_[{!s}] /; {!s}'.format(self.symbol_type.__name__, self.constraint)
        return '_[{!s}]'.format(self.symbol_type.__name__)


VariableReplacement = Union[Tuple[Expression, ...], Multiset[Expression], Expression]


class Substitution(Dict[str, VariableReplacement]):
    """Special :class:`dict` for substitutions with nicer formatting.

    The key is a variable's name and the value the substitution for it.
    """

    def try_add_variable(self, variable: str, replacement: VariableReplacement) -> None:
        """Try to add the variable with its replacement to the substitution.

        This considers an existing replacement and will only succeed if the new replacement
        can be merged with the old replacement. Merging can occur if either the two replacements
        are equivalent. Replacements can also be merged if the old replacement for the variable was
        unordered (i.e. a :class:`~.Multiset`) and the new one is an equivalent ordered version of it:

        >>> subst = Substitution({'x': Multiset(['a', 'b'])})
        >>> subst.try_add_variable('x', ('a', 'b'))
        >>> print(subst)
        {x ↦ (a, b)}

        Args:
            variable:
                The name of the variable to add.
            replacement:
                The replacement for the variable.

        Raises:
            ValueError:
                if the variable cannot be merged because it conflicts with the existing
                substitution for the variable.
        """
        if variable not in self:
            self[variable] = replacement
        else:
            existing_value = self[variable]

            if isinstance(existing_value, tuple):
                if isinstance(replacement, Multiset):
                    if Multiset(existing_value) != replacement:
                        raise ValueError
                elif replacement != existing_value:
                    raise ValueError
            elif isinstance(existing_value, Multiset):
                compare_value = Multiset(isinstance(replacement, Expression) and [replacement] or replacement)
                if existing_value == compare_value:
                    if not isinstance(replacement, Multiset):
                        self[variable] = replacement
                else:
                    raise ValueError
            elif replacement != existing_value:
                raise ValueError

    def union_with_variable(self, variable: str, replacement: VariableReplacement) -> 'Substitution':
        """Try to create a new substitution with the given variable added.

        See :meth:`try_add_variable` for a version of this method that modifies the substitution
        in place.

        Args:
            variable:
                The name of the variable to add.
            replacement:
                The substitution for the variable.

        Returns:
            The new substitution with the variable added or merged.

        Raises:
            ValueError:
                if the variable cannot be merged because it conflicts with the existing
                substitution for the variable.
        """
        new_subst = Substitution(self)
        new_subst.try_add_variable(variable, replacement)
        return new_subst

    def extract_substitution(self, expression: Expression, pattern: Expression) -> bool:
        """Extract the variable substitution for the given pattern and expression.

        This assumes that expression and pattern already match when being considered as linear.
        Also, they both must be :term:`syntactic`, as sequence variables cannot be handled here.
        All that this method does is checking whether all the substitutions for the variables can be unified.
        Also, this method mutates the substitution and might even do so in case the unification fails.
        So, in case it returns ``False``, the substitution is invalid for the match.

        Args:
            expression:
                A :term:`syntactic` expression that matches the pattern.
            pattern:
                A :term:`syntactic` pattern that matches the expression.

        Returns:
            ``True`` iff the substitution could be extracted successfully.
        """
        if isinstance(pattern, Variable):
            try:
                self.try_add_variable(pattern.name, expression)
            except ValueError:
                return False
            return self.extract_substitution(expression, pattern.expression)
        elif isinstance(pattern, Operation):
            assert isinstance(expression, type(pattern))
            assert len(expression.operands) == len(pattern.operands)
            op_expression = cast(Operation, expression)
            for expr, patt in zip(op_expression.operands, pattern.operands):
                if not self.extract_substitution(expr, patt):
                    return False
        return True

    def union(self, *others: 'Substitution') -> 'Substitution':
        """Try to merge the substitutions.

        If a variable occurs in multiple substitutions, try to merge the replacements.
        See :meth:`union_with_variable` to see how replacements are merged.

        >>> subst1 = Substitution({'x': Multiset(['a', 'b'])})
        >>> subst2 = Substitution({'x': ('a', 'b'), 'y': ('c', )})
        >>> print(subst1.union(subst2))
        {x ↦ (a, b), y ↦ (c)}

        Args:
            others:
                The other substitutions to merge with this one.

        Returns:
            The new substitution with the other substitutions merged.

        Raises:
            ValueError:
                if a variable occurs in multiple substitutions but cannot be merged because the
                substitutions conflict.
        """
        new_subst = Substitution(self)
        for other in others:
            for variable, replacement in other.items():
                new_subst.try_add_variable(variable, replacement)
        return new_subst

    def rename(self, renaming):
        return Substitution((renaming.get(name, name), value) for name, value in self.items())

    @staticmethod
    def _match_value_repr_str(value: Union[List[Expression], Expression]) -> str:  # pragma: no cover
        if isinstance(value, (list, tuple)):
            return '({!s})'.format(', '.join(str(x) for x in value))
        return str(value)

    def __str__(self):
        return '{{{}}}'.format(
            ', '.join('{!s} ↦ {!s}'.format(k, self._match_value_repr_str(v)) for k, v in sorted(self.items()))
        )

    def __repr__(self):
        return '{{{}}}'.format(', '.join('{!r}: {!r}'.format(k, v) for k, v in sorted(self.items())))


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
                attributes_to_copy = _frozen_type_cache[base][1]
                return FrozenExpression.__new__(cls, base(*args, **kwargs), attributes_to_copy)
        assert False, "unreachable, unless an invalid frozen operation subclass was created manually"

    def __repr__(cls):
        return cls._repr('FrozenOperation')


class FrozenExpression(Expression, metaclass=_FrozenMeta):
    """Base class for :term:`frozen` expressions.

    DO NOT instantiate this class directly, use :func:`freeze` instead!
    Only use this class for :func:`isinstance` checks.
    """
    # pylint: disable=assigning-non-slot

    __slots__ = ()

    # These are only added in the subclasses, because otherwise they would conflict with the ones of the
    # Expression (subclasses)
    _actual_slots = (
        '_frozen', 'cached_variables', 'cached_symbols', 'cached_is_constant', 'cached_is_syntactic',
        'cached_is_linear', 'cached_without_constraints', '_hash'
    )

    def __new__(cls, expr: Expression, attributes_to_copy: Iterable[str]) -> 'FrozenExpression':
        # This is a bit of a hack...
        # Copy all attributes of the original expression over to this instance
        # Use the _frozen attribute to lock changing attributes after the copying is done
        self = Expression.__new__(cls)

        object.__setattr__(self, '_frozen', False)
        self.constraint = expr.constraint

        if isinstance(expr, Operation):
            self.operands = tuple(freeze(e) for e in expr.operands)
            self.head = expr.head
        elif isinstance(expr, Symbol):
            self.name = expr.name
            self.head = self
        elif isinstance(expr, Variable):
            self.name = expr.name
            self.expression = freeze(expr.expression)
            self.head = self.expression.head
        elif isinstance(expr, Wildcard):
            self.min_count = expr.min_count
            self.fixed_size = expr.fixed_size
            self.head = None
            if isinstance(expr, SymbolWildcard):
                self.symbol_type = expr.symbol_type
        else:
            assert False, "Unreachable, unless new types of expressions are added which are not supported"

        for attribute in attributes_to_copy:
            setattr(self, attribute, getattr(expr, attribute))

        if hasattr(expr, '__dict__'):
            for attribute in expr.__dict__:
                if attribute != '__dict__':
                    setattr(self, attribute, getattr(expr, attribute))

        object.__setattr__(self, '_frozen', True)

        return self

    def __init__(self, expr, attributes_to_copy):  # pylint: disable=super-init-not-called
        pass

    def __setattr__(self, name, value):
        if self._frozen:  # pylint: disable=no-member
            raise TypeError("Cannot modify a FrozenExpression")
        else:
            object.__setattr__(self, name, value)

    @slot_cached_property('cached_variables')
    def variables(self) -> Multiset[str]:
        """Cached version of :attr:`Expression.variables`."""
        return super().variables

    @slot_cached_property('cached_symbols')
    def symbols(self) -> Multiset[str]:
        """Cached version of :attr:`Expression.symbols`."""
        return super().symbols

    @slot_cached_property('cached_is_constant')
    def is_constant(self) -> bool:
        """Cached version of :attr:`Expression.is_constant`."""
        return super().is_constant

    @slot_cached_property('cached_is_syntactic')
    def is_syntactic(self) -> bool:
        """Cached version of :attr:`Expression.is_syntactic`."""
        return super().is_syntactic

    @slot_cached_property('cached_is_linear')
    def is_linear(self) -> bool:
        """Cached version of :attr:`Expression.is_linear`."""
        return super().is_linear

    @slot_cached_property('cached_without_constraints')
    def without_constraints(self) -> 'FrozenExpression':
        """Cached version of :attr:`Expression.without_constraints`."""
        return freeze(super().without_constraints)

    def with_renamed_vars(self, renaming) -> 'FrozenExpression':
        return freeze(super().with_renamed_vars(renaming))

    def __hash__(self):
        # pylint: disable=no-member
        if not hasattr(self, '_hash'):
            object.__setattr__(self, '_hash', self._compute_hash())
        return self._hash


_frozen_type_cache = {}


def _frozen_new(cls, *args, **kwargs):
    return freeze(cls._original_base(*args, **kwargs))


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
                '__slots__': FrozenExpression._actual_slots,
                '__new__': _frozen_new
            }
        )  # yapf: disable

        attributes_to_copy = set()
        if hasattr(type(expression), '__slots__'):
            attributes_to_copy.update(
                *(getattr(cls, '__slots__', []) for cls in base.__mro__ if cls.__module__ != __name__)
            )
        _frozen_type_cache[base] = (frozen_class, attributes_to_copy)
    else:
        frozen_class, attributes_to_copy = _frozen_type_cache[base]

    return FrozenExpression.__new__(frozen_class, expression, attributes_to_copy)


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
        return expression._original_base(expression.name)
    if isinstance(expression, Operation):
        return expression._original_base.from_args(
            *map(unfreeze, expression.operands), constraint=expression.constraint
        )
    assert False, "Unreachable, unless new types of expressions are added that are unsupported"
