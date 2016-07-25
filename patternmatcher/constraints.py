# -*- coding: utf-8 -*-
from typing import Dict, Callable, Tuple
import inspect

from patternmatcher.expressions import Expression
from patternmatcher.utils import get_lambda_source

Match = Dict[str, Tuple[Expression]]

class Constraint(object):
    def __call__(self, match: Match) -> bool:
        raise NotImplementedError()

class MultiConstraint(Constraint):
    def __init__(self, *constraints: Constraint) -> None:
        self.constraints = list(constraints)

    def __call__(self, match: Match) -> bool:
        return all(c(match) for c in self.constraints)

    def __str__(self):
        return '(%s)' % ' and '.join(map(str, self.constraints))

class EqualVariablesConstraint(Constraint):
    def __init__(self, *variables: str) -> None:
        self.variables = list(variables)

    def __call__(self, match: Match) -> bool:
        v1 = self.variables[0]
        return all(match[v1] == match[v2] for v2 in self.variables[1:])

    def __str__(self):
        return '(%s)' % ' == '.join(self.variables)

class CustomConstraint(Constraint):
    def __init__(self, constraint: Callable[..., bool]) -> None:
        self.constraint = constraint
        signature = inspect.signature(constraint)

        self.allow_any = False
        self.variables = set()

        for param in signature.parameters.values():
            if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD or param.kind == inspect.Parameter.KEYWORD_ONLY:
                self.variables.add(param.name)
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                self.allow_any = True
            elif param.kind == inspect.Parameter.POSITIONAL_ONLY:
                raise ValueError('constraint cannot have positional-only arguments')

    def __call__(self, match: Match) -> bool:
        if self.allow_any:
            return self.constraint(**match)
        args = dict((name, match[name]) for name in self.variables)

        return self.constraint(**args)

    def __str__(self):
        return '(%s)' % get_lambda_source(self.constraint)


if __name__ == '__main__':
    cc = CustomConstraint(lambda x, y: x == y)
    print(cc.variables)
    print(cc({'x': 1, 'y': 1, 'z': 5, 'k': 8}))