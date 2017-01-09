# -*- coding: utf-8 -*-
"""Contains several pattern constraint classes.

A pattern constraint is used to further filter which subjects a pattern matches.

The most common use would be the :class:`CustomConstraint`, which wraps a lambda or function to act as a constraint:

>>> a_symbol_constraint = CustomConstraint(lambda x: x.name.startswith('a'))
>>> pattern = Variable.dot('x', constraint=a_symbol_constraint)
>>> is_match(Symbol('a1'), pattern)
True
>>> is_match(Symbol('b1'), pattern)
False

There is also the :class:`EqualVariablesConstraint` which will try to unify the substitutions of the variables and only
match if it succeeds:

>>> equal_constraint = EqualVariablesConstraint('x', 'y')
>>> pattern = f(x_, Variable.dot('y', constraint=equal_constraint))
>>> is_match(f(a, a), pattern)
True
>>> is_match(f(a, b), pattern)
False

Then there is the :class:`MultiConstraint`, that allows to combine multiple constraints:

>>> multi_constraint = MultiConstraint(equal_constraint, a_symbol_constraint)
>>> pattern = f(x_, y_, constraint=multi_constraint)
>>> is_match(f(a, a), pattern)
True
>>> is_match(f(a, Symbol('a2')), pattern)
False
>>> is_match(f(b, b), pattern)
False

You can also create a subclass of the :class:`Constraint` class to create your own custom constraint type.
"""
import inspect
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Callable, Dict, Optional

from . import substitution
from ..utils import get_short_lambda_source

__all__ = ['Constraint', 'MultiConstraint', 'EqualVariablesConstraint', 'CustomConstraint']


class Constraint(object, metaclass=ABCMeta):  # pylint: disable=too-few-public-methods
    """Base for pattern constraints.

    A constraint is essentially a callback, that receives the match :class:`Substitution` and returns a :class:`bool`
    indicating whether the match is valid.

    You have to override all the abstract methods if you wish to create your own subclass.
    """

    @abstractmethod
    def __call__(self, match: substitution.Substitution) -> bool:  # pylint: disable=missing-raises-doc
        """Return True, iff the constraint is fulfilled by the substitution.

        Override this in your subclass to define the actual constraint behavior.

        Args:
            match:
                The (current) match substitution. Note that the matching is done from left to right, so not all
                variables may have a value yet.

        Returns:
            True, iff the constraint is fulfilled by the substitution.
        """
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other):
        """Constraints need to be equatable."""
        raise NotImplementedError

    @abstractmethod
    def __hash__(self):
        """Constraints need to be hashable."""
        raise NotImplementedError

    @abstractmethod
    def with_renamed_vars(self, renaming: Dict[str, str]) -> 'Constraint':  # pylint: disable=missing-raises-doc
        """Return a *copy* of the constraint with renamed variables.

        This is called when the variables in the expression are renamed and hence the ones in the constraint have to be
        renamed as well. A later invocation of :meth:`__call__` will have the new variable names.

        You will have to implement this if your constraint needs to use the variables of the match substitution.
        Note that this can be called multiple times and you might have to account for that.
        Also, this should not modify the original constraint but rather return a copy.

        Args:
            renaming:
                A dictionary mapping old names to new names.

        Returns:
            A copy of the constraint with renamed variables.
        """
        raise NotImplementedError


class MultiConstraint(Constraint):
    """A constraint that combines multiple constraints into one.

    Use :meth:`create` instead of the constructor, because it will flatten the
    constraints and filter out ``None`` constraints.
    """

    def __init__(self, *constraints: Constraint) -> None:
        """
        Use :meth:`create` to construct a :class:`MultiConstraint` instead.

        Args:
            *constraints: The set of constraints.
        """
        self.constraints = frozenset(constraints)

    @classmethod
    def create(cls, *constraints: Optional[Constraint]) -> Optional[Constraint]:
        """Create a combine constraint from the given constraints.

        Nested MultiConstraints will be flattened:

        >>> constraint1 = CustomConstraint(lambda y: y < 1)
        >>> constraint2 = CustomConstraint(lambda x: x > 5)
        >>> constraint3 = CustomConstraint(lambda x, y: x != y)
        >>> MultiConstraint.create(MultiConstraint(constraint1, constraint2), constraint3)
        MultiConstraint(CustomConstraint(x != y) and CustomConstraint(x > 5) and CustomConstraint(y < 1))

        Also, ``None`` constraints are filtered out:

        >>> MultiConstraint.create(constraint1, None)
        CustomConstraint(y < 1)
        >>> MultiConstraint.create(None, None) is None
        True

        Args:
            *constraints:
                The set of constraints to combine. Constraints are filtered out if they are ``None``.

        Returns:
            The combined constraints. If it is a single constraint, it is returned, otherwise a :class:`MultiConstraint`
            combining the constraints is returned. If all constraints are ``None`` or none are given, then ``None`` is
            returned.
        """
        flat_constraints = set()
        for constraint in constraints:
            if isinstance(constraint, MultiConstraint):
                flat_constraints.update(constraint.constraints)
            elif constraint is not None:
                flat_constraints.add(constraint)

        if len(flat_constraints) == 1:
            return flat_constraints.pop()
        elif len(flat_constraints) == 0:
            return None

        return cls(*flat_constraints)

    def with_renamed_vars(self, renaming):
        return MultiConstraint(*(c.with_renamed_vars(renaming) for c in self.constraints))

    def __call__(self, match: substitution.Substitution) -> bool:
        return all(c(match) for c in self.constraints)

    def __str__(self):
        return '({!s})'.format(' and '.join(sorted(map(str, self.constraints))))

    def __repr__(self):
        return 'MultiConstraint({!s})'.format(' and '.join(sorted(map(repr, self.constraints))))

    def __eq__(self, other):
        return isinstance(other, MultiConstraint) and self.constraints == other.constraints

    def __hash__(self):
        return hash(self.constraints)


class EqualVariablesConstraint(Constraint):  # pylint: disable=too-few-public-methods
    """A constraint that ensure multiple variables are equal.

    The constraint tries to unify the substitutions for the variables and is fulfilled iff that succeeds.
    """

    def __init__(self, *variables: str) -> None:
        """
        Args:
            *variables: The names of the variables to check for equality.
        """
        self.variables = frozenset(variables)

    def with_renamed_vars(self, renaming):
        return EqualVariablesConstraint(*(renaming.get(v, v) for v in self.variables))

    def __call__(self, match: substitution.Substitution) -> bool:
        subst = substitution.Substitution()
        for name in self.variables:
            try:
                subst.try_add_variable('_', match[name])
            except ValueError:
                return False
        return True

    def __str__(self):
        return '({!s})'.format(' == '.join(sorted(self.variables)))

    def __repr__(self):
        return 'EqualVariablesConstraint({!s})'.format(' == '.join(sorted(self.variables)))

    def __eq__(self, other):
        return isinstance(other, EqualVariablesConstraint) and self.variables == other.variables

    def __hash__(self):
        return hash(self.variables)


class CustomConstraint(Constraint):  # pylint: disable=too-few-public-methods
    """Wrapper for lambdas of functions as constraints.

    The parameter names have to be the same as the the variable names in the expression:

    >>> constraint = CustomConstraint(lambda x, y: x.name < y.name)
    >>> pattern = f(x_, y_, constraint=constraint)
    >>> is_match(f(a, b), pattern)
    True
    >>> is_match(f(b, a), pattern)
    False

    The ordering of the parameters is not important. You only need to have the parameters needed for the constraint,
    not all variables occurring in the pattern.

    Note, that the matching happens from left left to right, so not all variables may have been assigned a value when
    constraint is called. For constraints over multiple variables you should attach the constraint to the last
    variable occurring in the pattern or a surrounding operation.
    """

    def __init__(self, constraint: Callable[..., bool]) -> None:
        """
        Args:
            constraint:
                The constraint callback.

        Raises:
            ValueError:
                If the callback has positional-only or variable parameters (*args and **kwargs).
        """
        self.constraint = constraint
        signature = inspect.signature(constraint)

        self.variables = OrderedDict()

        for param in signature.parameters.values():
            if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD or param.kind == inspect.Parameter.KEYWORD_ONLY:
                self.variables[param.name] = param.name
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                raise ValueError("Constraint cannot have variable keyword arguments ({})".format(param.name))
            else:
                raise ValueError(
                    "Constraint cannot have positional-only or variable positional arguments ({})".format(param.name)
                )

    def with_renamed_vars(self, renaming):
        cc = CustomConstraint(self.constraint)
        for param_name, old_name in list(cc.variables.items()):
            cc.variables[param_name] = renaming.get(old_name, old_name)
        return cc

    def __call__(self, match: substitution.Substitution) -> bool:
        args = dict((name, match[var_name]) for name, var_name in self.variables.items())

        return self.constraint(**args)

    def __str__(self):
        return '({!s})'.format(get_short_lambda_source(self.constraint) or self.constraint.__name__)

    def __repr__(self):
        return 'CustomConstraint({!s})'.format(get_short_lambda_source(self.constraint) or self.constraint.__name__)

    def __eq__(self, other):
        return (
            isinstance(other, CustomConstraint) and self.constraint == other.constraint and
            self.variables == other.variables
        )

    def __hash__(self):
        return hash(self.constraint)
