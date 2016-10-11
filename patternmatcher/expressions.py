# -*- coding: utf-8 -*-
import itertools
import keyword
from enum import Enum, EnumMeta
from typing import (Callable, Dict, Iterator, List, NamedTuple, Optional,
                    Set, Tuple, TupleMeta, Type, Union)

from patternmatcher.multiset import Multiset
from patternmatcher.utils import cached_property


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

    nullary     = (0, True)
    unary       = (1, True)
    binary      = (2, True)
    ternary     = (3, True)
    polyadic    = (2, False)
    variadic    = (0, False)

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

class _OperationMeta(type):
    def __repr__(cls):
        return 'Operation[%r, arity=%r, associative=%r, commutative=%r, one_identity=%r]' % \
            (cls.name, cls.arity, cls.associative, cls.commutative, cls.one_identity)

    def __call__(cls, *operands: Expression, constraint:Optional[Constraint]=None):
        # __call__ is overriden, so that for one_identity operations with a single argument
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
            new_operands = [] # type: List[Expression]
            for operand in operands:
                if isinstance(operand, cls):
                    new_operands.extend(operand.operands) # type: ignore
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


class Operation(Expression, metaclass=_OperationMeta):
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

    def __init__(self, *operands: Expression, constraint: Optional[Constraint]=None) -> None:
        """Base class for all expressions.

        All expressions are immutable, i.e. their attributes should not be changed,
        as several attributes are computed at instantiation and are not refreshed.

        Args:
            operands
                The operands for the operation expression.
            constraint
                An optional constraint expression, which is checked for each match
                to verify it.

        Raises:
            ValueError: if the operand count does not match the operation's arity.
        """
        super().__init__(constraint)

        if len(operands) < self.arity.min_count:
            raise ValueError("Operation %s got arity %s, but got %d operands." % (self.__class__.__name__, self.arity, len(operands)))

        if self.arity.fixed_size and len(operands) > self.arity.min_count:
            msg = "Operation %s got arity %s, but got %d operands." % (self.__class__.__name__, self.arity, len(operands))
            if self.associative:
                msg += " Associative operations should have a variadic/polyadic arity."
            raise ValueError(msg)

        variables = dict()
        for var, _ in itertools.chain.from_iterable(o.preorder_iter(lambda e: isinstance(e, Variable)) for o in operands):
            if var.name in variables:
                if variables[var.name] != var:
                    raise ValueError("Conflicting versions of variable %s: %r vs %r" % (var.name, var, variables[var.name]))
            else:
                variables[var.name] = var

        self.operands = list(operands)
        self.head = type(self)

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

        >>> Times = Operation.new('*', Arity.polyadic, 'Times', associative=True, commutative=True, one_identity=True)
        >>> Times
        Operation['*', arity=Arity.polyadic, associative=True, commutative=True, one_identity=True]
        >>> str(Times(Symbol('a'), Symbol('b')))
        '*(a, b)'

        Args:
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

        Raises:
            ValueError: if the class name of the operation is not a valid class identifier.
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
        return isinstance(self, other.__class__) and \
               self.constraint == other.constraint and \
               len(self.operands) == len(other.operands) and \
               all(x == y for x,y in zip(self.operands, other.operands))

    def __getitem__(self, key):
        if len(key) == 0:
            return self
        head, *remainder = key
        return self.operands[head][remainder]

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

    def _compute_hash(self):
        return hash((type(self), ) + tuple(self.operands))

class Atom(Expression):
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
        return isinstance(self, other.__class__) and self.name == other.name

    def _compute_hash(self):
        return hash((type(self), self.name))

class Variable(Expression):
    """A variable that is captured during a match.

    Wraps another pattern expression that is used to match. On match, the matched
    value is captured for the variable.
    """

    def __init__(self, name: str, expression: Expression, constraint:Optional[Constraint]=None) -> None:
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
        """Create a :class:`Variable` with a :class:`Wildcard` that matches exactly one argument."""
        return Variable(name, Wildcard.dot(), constraint)

    @staticmethod
    def symbol(name: str, symbol_type:Type[Symbol]=Symbol, constraint:Optional[Constraint]=None):
        """Create a :class:`Variable` with a :class:`SymbolWildcard`.

        Args:
            name:
                The name of the variable.
            symbol_type:
                An optional subclass of :class:`Symbol` to further limit which kind of smybols are
                matched by the wildcard.
            constraint:
                An optional :class:`.Constraint` which can filter wwhat is matched by the variable.

        Returns:
            A :class:`Variable` that matches a :class:`Symbol` with type ``symbol_type``.
        """
        return Variable(name, Wildcard.symbol(symbol_type), constraint)

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

    def _compute_hash(self):
        return hash((type(self), self.name, self.expression))


class Wildcard(Atom):
    """A wildcard that matches any expression.

    The wildcard will match any number of expressions between `min_count` and `fixed_size`.
    Optionally, the wildcard can also be constrained to only match expressions satisfying a predicate.
    """

    def __init__(self, min_count: int, fixed_size: bool, constraint: Optional[Constraint]=None) -> None:
        """
        Args:
            min_count
                The minimum number of expressions this wildcard will match. Must be a non-negative number.
            fixed_size
                If `True`, the wildcard matches exactly `min_count` expressions.
                If `False`, the wildcard is a sequence wildcard and can match `min_count` or more expressions.
            constraint
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
    def symbol(symbol_type:Type[Symbol]=Symbol):
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

    def _compute_hash(self):
        return hash((type(self), self.min_count, self.fixed_size))


class SymbolWildcard(Wildcard):
    """A special :class:`Wildcard` that matches a :class:`Symbol`."""

    def __init__(self, symbol_type:Type[Symbol]=Symbol, constraint: Optional[Constraint]=None) -> None:
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

    def __repr__(self):
        if self.constraint:
            return '%s(%r, constraint=%r)' % (self.__class__.__name__, self.symbol_type, self.constraint)
        return '%s(%r)' % (self.__class__.__name__, self.symbol_type)


VariableReplacement = Union[Tuple[Expression], Set[Expression], Expression]

class Substitution(Dict[str, VariableReplacement]):
    """Special :class:`dict` for substitutions with nicer formatting.

    The key is a variable's name and the value the substitution for it.
    """

    def try_add_variable(self, variable: str, replacement: VariableReplacement):
        """Try to add the variable with its replacement to the substitution.

        This considers an existing replacement and will only succeed if the new replacement
        can be merged with the old replacement. Merging can occur if either the two replacements
        are equivalent. Replacements can also be merged if the old replacement for the variable was
        unordered (i.e. a :class:`~typing.Set`) and the new one is an equivalant ordered version of it:

        >>> subst = Substitution({'x': {'a', 'b'}})
        >>> subst.try_add_variable('x', ('a', 'b'))
        >>> subst
        {'x': ('a', 'b')}

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
                if isinstance(replacement, set):
                    if Multiset(existing_value) != Multiset(replacement):
                        raise ValueError
                elif replacement != existing_value:
                    raise ValueError
            elif isinstance(existing_value, Set):
                compare_value = Multiset(isinstance(replacement, Expression) and [replacement] or replacement)
                if existing_value == compare_value:
                    if not isinstance(replacement, Set):
                        self[variable] = replacement
                else:
                    raise ValueError
            elif replacement != existing_value:
                raise ValueError

    def union_with_variable(self, variable: str, replacement: VariableReplacement):
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

    def union(self, *others: 'Substitution'):
        """Try to merge the substitutions.

        If a variable occurs in multiple substitutions, try to merge the replacements.
        See :meth:`union_with_variable` to see how replacements are merged.

        >>> subst1 = Substitution({'x': {'a', 'b'}})
        >>> subst2 = Substitution({'x': ('a', 'b'), 'y': ('c', )})
        >>> str(subst1.union(subst2))
        'x ← (a, b), y ← (c)'

        Args:
            others:
                The other substitutions to merge with this one.

        Returns:
            The new substitution with the other substitutions merged.

        Raises:
            ValueError:
                if a variable occures in multiple substitutions but cannot be merged because the
                substitutions conflict.
        """
        new_subst = Substitution(self)
        for other in others:
            for variable, replacement in other.items():
                new_subst.try_add_variable(variable, replacement)
        return new_subst

    @staticmethod
    def _match_value_repr_str(value: Union[List[Expression], Expression]) -> str: # pragma: no cover
        if isinstance(value, (list, tuple)):
            return '(%s)' % (', '.join(str(x) for x in value))
        return str(value)

    def __str__(self):
        return ', '.join('%s ← %s' % (k, self._match_value_repr_str(v)) for k, v in sorted(self.items()))

class _FrozenMeta(type):
    __call__ = type.__call__

class _FrozenOperationMeta(_FrozenMeta, _OperationMeta):
    pass

class FrozenExpression(Expression, metaclass=_FrozenMeta):
    def __new__(cls, expr: Expression):
        self = Expression.__new__(cls)
        object.__setattr__(self, '_frozen', False)
        self.constraint = expr.constraint

        if isinstance(expr, Operation):
            self.operands = tuple(freeze(e) for e in expr.operands)
            self.head = type(self)
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

        object.__setattr__(self, '_frozen', True)

        return self

    def __init__(self, expr): # pylint: disable=super-init-not-called
        pass

    def __setattr__(self, name, value):
        if self._frozen: # pylint: disable=no-member
            raise TypeError('Cannot modifiy a FrozenExpression')
        else:
            object.__setattr__(self, name, value)

    @cached_property
    def variables(self) -> Multiset:
        return super().variables

    @cached_property
    def symbols(self) -> Multiset:
        return super().symbols

    @cached_property
    def is_constant(self) -> Multiset:
        return super().is_constant

    @cached_property
    def is_syntactic(self) -> Multiset:
        return super().is_syntactic

    @cached_property
    def is_linear(self) -> Multiset:
        return super().is_linear

    def __hash__(self):
        # pylint: disable=no-member
        if not hasattr(self, '_hash'):
            object.__setattr__(self, '_hash', self._compute_hash())
        return self._hash


_frozen_type_cache = {}

def freeze(expr: Expression) -> FrozenExpression:
    base = type(expr)
    if base not in _frozen_type_cache:
        meta = isinstance(base, _OperationMeta) and _FrozenOperationMeta or _FrozenMeta
        _frozen_type_cache[base] = meta('Frozen' + base.__name__, (FrozenExpression, base), {})
    return _frozen_type_cache[base](expr)

def unfreeze(expr: FrozenExpression) -> Expression:
    # TODO
    return expr

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
