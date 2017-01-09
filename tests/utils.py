# -*- coding: utf-8 -*-
from matchpy.expressions import Constraint

class MockConstraint(Constraint):
    def __init__(self, return_value):
        self.return_value = return_value
        self.called_with = []
        self.renaming = None

    def __call__(self, match):
        self.called_with.append(match)
        return self.return_value

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))

    def __repr__(self):
        return 'MockConstraint(%r)' % self.return_value

    def with_renamed_vars(self, renaming):
        self.renaming = renaming
        return self

    @property
    def call_count(self):
        return len(self.called_with)

    def assert_called_with(self, args):
        if self.renaming is not None:
            args = dict((self.renaming.get(n, n), v) for n, v in args.items())
        assert args in self.called_with