# -*- coding: utf-8 -*-
from matchpy.expressions.constraints import Constraint
from matchpy.expressions.substitution import Substitution
from matchpy.expressions.expressions import Pattern


class MockConstraint(Constraint):
    def __init__(self, return_value, *variables, renaming=None):
        self.return_value = return_value
        self.called_with = []
        self._variables = set(variables)
        self.renaming = renaming or {}

    def __call__(self, match):
        self.called_with.append(Substitution(match))
        return self.return_value

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))

    def __repr__(self):
        if self.variables:
            return 'MockConstraint({!r}, {}, renaming={!r})'.format(
                self.return_value, ', '.join(map(repr, self.variables)), self.renaming)
        return 'MockConstraint({!r}, renaming={!r})'.format(self.return_value, self.renaming)

    def with_renamed_vars(self, renaming):
        self.renaming.update(renaming)
        return self

    @property
    def variables(self):
        return set(self.renaming.get(v, v) for v in self._variables)

    @property
    def call_count(self):
        return len(self.called_with)

    def assert_called_with(self, args):
        args = dict((self.renaming.get(n, n), v) for n, v in args.items())
        assert args in self.called_with, "Constraint was not called with {}. List of calls: {}".format(
            args, self.called_with
        )


def assert_match_as_expected(match, subject, pattern, expected_matches):
    pattern = Pattern(pattern)
    matches = list(match(subject, pattern))
    assert len(matches) == len(expected_matches), 'Unexpected number of matches'
    for expected_match in expected_matches:
        assert expected_match in matches, "Subject {!s} and pattern {!s} did not yield the match {!s} but were supposed to".format(
            subject, pattern, expected_match
        )
    for match in matches:
        assert match in expected_matches, "Subject {!s} and pattern {!s} yielded the unexpected match {!s}".format(
            subject, pattern, match
        )
