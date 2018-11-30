# -*- coding: utf-8 -*-
from hypothesis import assume, given
import hypothesis.strategies as st
import pytest
from multiset import Multiset

from matchpy.expressions.constraints import CustomConstraint
from matchpy.expressions.expressions import Symbol, Wildcard, Pattern
from matchpy.matching.many_to_one import ManyToOneMatcher
from matchpy.functions import substitute
from .utils import MockConstraint
from .common import *

@pytest.mark.parametrize(
    '   expression,             pattern,            expected_matches',
    [
        ({'b': 0},              {'a': x_},          []),
        (('a', 0),              {'a': x_},          []),
        ({'a': 0},              {'a': x_},          [{'x': 0}]),
        ({'a': 0},              {x_: 0},            [{'x': 'a'}]),
        ({'a': 0, 'b': 1},      {x_: 0, _: _},      [{'x': 'a'}]),
        ({'a': 0, 'b': 0},      {x_: 0, _: _},      [{'x': 'a'}, {'x': 'b'}]),
        ({'a': 0, 'b': 0},      {'a': _, 'b': _},   [{}]),
        ({'a': 0, 'b': 0},      {'a': _, 'c': _},   []),
    ]
)  # yapf: disable
def test_dict_match(match, expression, pattern, expected_matches):
    expression = expression
    pattern = Pattern(pattern)
    result = list(match(expression, pattern))
    for expected_match in expected_matches:
        assert expected_match in result, "Expression {!s} and {!s} did not yield the match {!s} but were supposed to".format(
            expression, pattern, expected_match
        )
    for result_match in result:
        assert result_match in expected_matches, "Expression {!s} and {!s} yielded the unexpected match {!s}".format(
            expression, pattern, result_match
        )
