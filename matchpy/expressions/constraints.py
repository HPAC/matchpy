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

You can also create a subclass of the :class:`Constraint` class to create your own custom constraint type.
"""
import inspect
from collections import OrderedDict
from typing import Callable, Optional, FrozenSet

from . import substitution
from ..utils import get_short_lambda_source, cached_property

__all__ = ['Constraint', 'EqualVariablesConstraint', 'CustomConstraint']


class Constraint(object):  # pylint: disable=too-few-public-methods
    """Base for pattern constraints.

    A constraint is essentially a callback, that receives the match :class:`Substitution` and returns a :class:`bool`
    indicating whether the match is valid.

    You have to override all the abstract methods if you wish to create your own subclass.
    """

    def __call__(self, match: substitution.Substitution) -> bool:  # pylint: disable=missing-raises-doc
        """Return True, iff the constraint is fulfilled by the substitution.

        Override this in your subclass to define the actual constraint behavior.

        Args:
            match:
                The (current) match substitution. Note that the matching is done from left to right, so not all
                variables may have a value yet. You need to override `variables` so that the constraint gets
                called once all the variables it depends on have a value assigned to them.

        Returns:
            True, iff the constraint is fulfilled by the substitution.
        """
        raise NotImplementedError

    def __eq__(self, other):
        """Constraints need to be equatable."""
        raise NotImplementedError

    def __hash__(self):
        """Constraints need to be hashable."""
        raise NotImplementedError

    @property
    def variables(self) -> FrozenSet[str]:
        """The names of the variables the constraint depends upon.

        Used by matchers to decide when a constraint can be evaluated (which is when all
        the dependency variables have been assigned a value). If the set is empty, the constraint will
        only be evaluated once the whole match is complete.
        """
        return frozenset()


class EqualVariablesConstraint(Constraint):  # pylint: disable=too-few-public-methods
    """A constraint that ensure multiple variables are equal.

    The constraint tries to unify the substitutions for the variables and is fulfilled iff that succeeds.
    """

    def __init__(self, *variables: str) -> None:
        """
        Args:
            *variables: The names of the variables to check for equality.
        """
        self._variables = frozenset(variables)

    @property
    def variables(self):
        return self._variables

    def __call__(self, match: substitution.Substitution) -> bool:
        subst = substitution.Substitution()
        for name in self._variables:
            try:
                subst.try_add_variable('_', match[name])
            except ValueError:
                return False
        return True

    def __str__(self):
        return '({!s})'.format(' == '.join(sorted(self._variables)))

    def __repr__(self):
        return 'EqualVariablesConstraint({!s})'.format(' == '.join(sorted(self._variables)))

    def __eq__(self, other):
        return isinstance(other, EqualVariablesConstraint) and self._variables == other._variables

    def __hash__(self):
        return hash(self._variables)


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

        self._variables = OrderedDict()

        for param in signature.parameters.values():
            if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD or param.kind == inspect.Parameter.KEYWORD_ONLY:
                self._variables[param.name] = param.name
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                raise ValueError("Constraint cannot have variable keyword arguments ({})".format(param.name))
            else:
                raise ValueError(
                    "Constraint cannot have positional-only or variable positional arguments ({})".format(param.name)
                )


    @property
    def variables(self):
        return frozenset(self._variables.keys())

    def __call__(self, match: substitution.Substitution) -> bool:
        args = dict((name, match[var_name]) for name, var_name in self._variables.items())

        return self.constraint(**args)

    def __str__(self):
        return '({!s})'.format(get_short_lambda_source(self.constraint) or self.constraint.__name__)

    def __repr__(self):
        return 'CustomConstraint({!s})'.format(get_short_lambda_source(self.constraint) or self.constraint.__name__)

    def __eq__(self, other):
        return (
            isinstance(other, CustomConstraint) and self.constraint == other.constraint and
            self._variables == other._variables
        )

    def __hash__(self):
        return hash(self.constraint)
