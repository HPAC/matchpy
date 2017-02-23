# -*- coding: utf-8 -*-
from matchpy.expressions.constraints import Constraint


class MockConstraint(Constraint):
    def __init__(self, return_value, *variables):
        self.return_value = return_value
        self.called_with = []
        self._variables = set(variables)

    def __call__(self, match):
        self.called_with.append(match)
        return self.return_value

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))

    def __repr__(self):
        return 'MockConstraint(%r, %r)' % (self.return_value, self._variables)

    @property
    def variables(self):
        return self._variables

    @property
    def call_count(self):
        return len(self.called_with)

    def assert_called_with(self, args):
        assert args in self.called_with, "Constraint was not called with {}. List of calls: {}".format(args, self.called_with)
