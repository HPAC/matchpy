# -*- coding: utf-8 -*-
"""Module contains the basic building blocks for expressions.

TODO: Document each class with an example here.
"""
import itertools
import keyword
from abc import ABCMeta
from enum import Enum, EnumMeta
from typing import List, NamedTuple, Set, Tuple, TupleMeta, Type, Optional  # pylint: disable=unused-import

from multiset import Multiset

from .base import Expression, ExprPredicate, ExpressionsWithPos, Constraint

__all__ = ['Arity', 'Atom', 'Symbol', 'Variable', 'Wildcard', 'Operation', 'SymbolWildcard']


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
    new operation expression. This is done to avoid problems when overriding ``__new__`` of the operation class
    and clashes with the `.FrozenOperation` class.
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
    """Base for all atomic expressions."""
    pass


class Symbol(Atom):
    """An atomic constant expression term.

    It is uniquely identified by its name.

    Attributes:
        name (str):
            The symbol's name.
    """

    __slots__ = 'name',

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
        return getattr(self, '_original_base', self.__class__)(self.name)

    def with_renamed_vars(self, renaming) -> 'Symbol':
        return getattr(self, '_original_base', self.__class__)(self.name)

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
            assigned to the variable. Usually, a `Wildcard` is used to match
            any expression.
        constraint (Optional[Constraint]):
            See :attr:`Expression.constraint`.

    """

    __slots__ = 'name', 'expression'

    def __init__(self, name: str, expression: Expression, constraint: Constraint=None) -> None:
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
        return Variable(self.name, self.expression.without_constraints)

    def with_renamed_vars(self, renaming) -> 'Variable':
        constraint = self.constraint.with_renamed_vars(renaming) if self.constraint else None
        name = renaming.get(self.name, self.name)
        return Variable(name, self.expression.with_renamed_vars(renaming), constraint)

    @staticmethod
    def dot(name: str, constraint: Constraint=None) -> 'Variable':
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
    def symbol(name: str, symbol_type: Type[Symbol]=Symbol, constraint: Constraint=None) -> 'Variable':
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
    def star(name: str, constraint: Constraint=None) -> 'Variable':
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
    def plus(name: str, constraint: Constraint=None) -> 'Variable':
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
    def fixed(name: str, length: int, constraint: Constraint=None) -> 'Variable':
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

    __slots__ = 'min_count', 'fixed_size'

    def __init__(self, min_count: int, fixed_size: bool, constraint: Constraint=None) -> None:
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
        return Wildcard(self.min_count, self.fixed_size)

    def with_renamed_vars(self, renaming) -> 'Wildcard':
        constraint = self.constraint.with_renamed_vars(renaming) if self.constraint else None
        return Wildcard(self.min_count, self.fixed_size, constraint)

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
    """A special `Wildcard` that matches a `Symbol`.

    Attributes:
        symbol_type:
            A subclass of `Symbol` to constraint what the wildcard matches.
            If not specified, the wildcard will match any `Symbol`.
    """

    __slots__ = 'symbol_type',

    def __init__(self, symbol_type: Type[Symbol]=Symbol, constraint: Constraint=None) -> None:
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
