# -*- coding: utf-8 -*-
import inspect
from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, Set

from .expressions import Substitution
from .utils import get_short_lambda_source


class Constraint(object, metaclass=ABCMeta):
    """Base for pattern constraints.
    """
    @abstractmethod
    def __call__(self, match: Substitution) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError()

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError()


class MultiConstraint(Constraint):
    def __init__(self, *constraints: Constraint) -> None:
        self.constraints = frozenset(constraints)

    @classmethod
    def create(cls, *constraints: Optional[Constraint]) -> Optional[Constraint]:
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

    def __call__(self, match: Substitution) -> bool:
        return all(c(match) for c in self.constraints)

    def __str__(self):
        return '({!s})'.format(' and '.join(map(str, self.constraints)))

    def __repr__(self):
        return 'MultiConstraint({!s})'.format(' and '.join(map(repr, self.constraints)))

    def __eq__(self, other):
        return isinstance(other, MultiConstraint) and self.constraints == other.constraints

    def __hash__(self):
        return hash(self.constraints)


class EqualVariablesConstraint(Constraint):
    def __init__(self, *variables: str) -> None:
        self.variables = frozenset(variables)

    def __call__(self, match: Substitution) -> bool:
        subst = Substitution()
        for name in self.variables:
            try:
                subst.try_add_variable('_', match[name])
            except ValueError:
                return False
        return True

    def __str__(self):
        return '({!s})'.format(' == '.join(self.variables))

    def __repr__(self):
        return 'EqualVariablesConstraint({!s})'.format(' == '.join(self.variables))

    def __eq__(self, other):
        return isinstance(other, EqualVariablesConstraint) and self.variables == other.variables

    def __hash__(self):
        return hash(self.variables)


class CustomConstraint(Constraint):
    def __init__(self, constraint: Callable[..., bool]) -> None:
        self.constraint = constraint
        signature = inspect.signature(constraint)

        self.allow_any = False
        self.variables = set()  # type: Set[str]

        for param in signature.parameters.values():
            if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD or param.kind == inspect.Parameter.KEYWORD_ONLY:
                self.variables.add(param.name)
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                self.allow_any = True
            else:
                raise ValueError("constraint cannot have positional-only or variable positional arguments (*args)")

    def __call__(self, match: Substitution) -> bool:
        if self.allow_any:
            return self.constraint(**match)
        args = dict((name, match[name]) for name in self.variables)

        return self.constraint(**args)

    def __str__(self):
        return '({!s})'.format(get_short_lambda_source(self.constraint) or self.constraint.__name__)

    def __repr__(self):
        return 'CustomConstraint({!s})'.format(get_short_lambda_source(self.constraint) or self.constraint.__name__)

    def __eq__(self, other):
        return isinstance(other, CustomConstraint) and self.constraint == other.constraint

    def __hash__(self):
        return hash(self.constraint)
