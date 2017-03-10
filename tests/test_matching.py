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


def _convert_match_list_to_tuple(expected_match):
    for var, val in expected_match.items():
        if isinstance(val, list):
            expected_match[var] = tuple(val)


class TestMatch:
    @pytest.mark.parametrize(
        '   expression,         pattern,        is_match',
        [
            (a,                 a,              True),
            (b,                 a,              False),
            (f(),               f(),            True),
            (f(a),              f(),            False),
            (f(a),              f(a),           True),
            (f(b),              f(a),           False),
            (f(),               f(a),           False),
            (f2(a),             f(a),           False),
            (f(a, b),           f(a),           False),
            (f(a, b),           f(a, b),        True),
            (f(a),              f(a, b),        False),
            (f(b, a),           f(a, b),        False),
            (f(a, b, c),        f(a, b),        False),
            (f(a, f2(b)),       f(a, b),        False),
            (f(f2(a), f2(b)),   f(a, b),        False),
            (f(f2(a), b),       f(a, b),        False),
            (f(f(a, b)),        f(a, b),        False),
            (f2(a, b),          f(a, b),        False),
            (f(a, f2(b)),       f(a, f2(b)),    True),
            (f(f2(a), b),       f(f2(a), b),    True),
            (f(f(a, b)),        f(f(a, b)),     True)
        ]
    )  # yapf: disable
    def test_constant_match(self, match_syntactic, expression, pattern, is_match):
        expression = expression
        pattern = Pattern(pattern)
        result = list(match_syntactic(expression, pattern))
        if is_match:
            assert result == [dict()], "Expression {!s} and {!s} did not match but were supposed to".format(
                expression, pattern
            )
        else:
            assert result == [], "Expression {!s} and {!s} did match but were not supposed to".format(
                expression, pattern
            )

    @pytest.mark.parametrize(
        '   expression,         pattern,            is_match',
        [
            (f_c(),              f_c(),               True),
            (f_c(a),             f_c(),               False),
            (f_c(),              f_c(a),              False),
            (f_c(a, b),          f_c(a, b),           True),
            (f_c(a, b),          f_c(a, b),           True),
            (f_c(b, a),          f_c(a, b),           True),
            (f_c(b, a, c),       f_c(a, b, c),        True),
            (f_c(c, a, b),       f_c(a, b, c),        True),
            (f_c(b, a, c),       f_c(c, b, a),        True),
            (f_c(b, a, a),       f_c(a, a, b),        True),
            (f_c(a, b, a),       f_c(a, a, b),        True),
            (f_c(b, b, a),       f_c(a, a, b),        False),
            (f_c(c, a, f2(b)),   f_c(a, f2(b), c),    True),
            (f_c(c, a, f2(b)),   f_c(f2(a), b, c),    False),
            (f2(c, f_c(a, b)),   f2(c, f_c(b, a)),    True),
            (f2(c, f_c(a, b)),   f2(f_c(a, b), c),    False),
            (f_c(c, f_c(a, b)),  f_c(f_c(a, b), c),   True),
            (f_c(c, f_c(b, a)),  f_c(f_c(a, b), c),   True),
            (f_c(c, f_c(b, b)),  f_c(f_c(a, b), c),   False),
            (f_c(a, f_c(b, a)),  f_c(f_c(a, b), c),   False),
        ]
    )  # yapf: disable
    def test_commutative_match(self, match, expression, pattern, is_match):
        expression = expression
        pattern = Pattern(pattern)
        result = list(match(expression, pattern))
        if is_match:
            assert result == [dict()], "Expression {!s} and {!s} did not match but were supposed to".format(
                expression, pattern
            )
        else:
            assert result == [], "Expression {!s} and {!s} did match but were not supposed to".format(
                expression, pattern
            )

    @pytest.mark.parametrize(
        '   expression,         pattern,                    match_count',
        [
            (f_c(f(a)),          f_c(f(x_)),                  1),
            (f_c(f(a), f(b)),    f_c(f(x_), f(y_)),           2),
            (f_c(f(a), f(b)),    f_c(f(x_), f(x_)),           0),
            (f_c(f(a), f(a)),    f_c(f(x_), f(x_)),           1),
            (f_c(f(a), f(b)),    f_c(f(x_), y_),              2),
            (f_c(f(a), f(a)),    f_c(f(x_), x_),              0),
            (f_c(f(a), a),       f_c(f(x_), x_),              1),
            (f_c(f(a), a),       f_c(f(x_)),                  0),
            (f_c(f(a), a),       f_c(f(x_), f(y_)),           0),
            (f_c(f(a), f(a)),    f_c(f(x_), f(y_), f(z_)),    0),
            (f_c(f2(a), f2(a)),  f_c(f(x_), f(y_)),           0),
            (f2_c(f(a),  f(a)),  f_c(f(x_), f(y_)),           0),
            (f_c(f_c(a, b), c),  f_c(f_c(x__), y__),          1),
        ]
    )  # yapf: disable
    def test_commutative_syntactic_match(self, match, expression, pattern, match_count):
        expression = expression
        pattern = Pattern(pattern)
        result = list(match(expression, pattern))
        assert len(result) == match_count, 'Wrong number of matches'

        for subst in result:
            assert substitute(pattern.expression, subst) == expression, 'Invalid match: {}'.format(subst)

    @pytest.mark.parametrize(
        '   expression,         pattern,    constraint,              expected_matches',
        [
            (f_c(a),             f_c(x_),   mock_constraint_false,   []),
            (f_c(a),             f_c(x_),   mock_constraint_true,    [{'x': a}]),
        ]
    )  # yapf: disable
    def test_commutative_constraint_match(self, match, expression, pattern, constraint, expected_matches):
        expression = expression
        pattern = Pattern(pattern, constraint)
        result = list(match(expression, pattern))
        for expected_match in expected_matches:
            assert expected_match in result, "Expression {!s} and {!s} did not yield the match {!s} but were supposed to".format(
                expression, pattern, expected_match
            )
        for result_match in result:
            assert result_match in expected_matches, "Expression {!s} and {!s} yielded the unexpected match {!s}".format(
                expression, pattern, result_match
            )

    @pytest.mark.parametrize(
        '   expression,         pattern,                            expected_match',
        [
            (a,                 x_,                                 {'x': a}),
            (b,                 x_,                                 {'x': b}),
            (f(a),              f(x_),                              {'x': a}),
            (f(b),              f(x_),                              {'x': b}),
            (f(a),              x_,                                 {'x': f(a)}),
            (f2(a),             f(x_),                              None),
            (f(a, b),           f(x_),                              None),
            (f(a, b),           f(x_, b),                           {'x': a}),
            (f(a, b),           f(x_, a),                           None),
            (f(a, b),           f(a, x_),                           {'x': b}),
            (f(a, b),           f(x_, x_),                          None),
            (f(a, a),           f(x_, x_),                          {'x': a}),
            (f(a, b),           f(x_, y_),                          {'x': a,       'y': b}),
            (f(a),              f(x_, y_),                          None),
            (f(a, b, c),        f(x_, y_),                          None),
            (f(a, f2(b)),       f(x_, y_),                          {'x': a,       'y': f2(b)}),
            (f(a, f2(b)),       f(x_, f2(y_)),                      {'x': a,       'y': b}),
            (f(a, f2(b)),       f(x_, f2(x_)),                      None),
            (f(a, f2(a)),       f(x_, f2(x_)),                      {'x': a}),
            (f(f2(a), f2(b)),   f(x_, x_),                          None),
            (f(f2(a), f2(b)),   f(x_, y_),                          {'x': f2(a),   'y': f2(b)}),
            (f(f2(a), a),       f(x_, x_),                          None),
            (f(f2(a), a),       f(f2(x_), x_),                      {'x': a}),
            (f(f(a, b)),        f(x_, y_),                          None),
            (f(f(a, b)),        f(x_),                              {'x': f(a, b)}),
            (f2(a, b),          f(x_, y_),                          None),
            (f(f(a, b)),        f(f(x_, y_)),                       {'x': a,       'y': b}),
          # (f(a, b, c),        f(x2_),                             None),
          # (f(a, b),           f(x2_),                             {'x': (a, b)}),
          # (f(a),              f(x2_),                             None),
        ]
    )  # yapf: disable
    def test_wildcard_dot_match(self, match_syntactic, expression, pattern, expected_match):
        expression = expression
        pattern = Pattern(pattern)
        result = list(match_syntactic(expression, pattern))
        if expected_match is not None:
            assert result == [expected_match
                             ], "Expression {!s} and {!s} did not match as {!s} but were supposed to".format(
                                 expression, pattern, expected_match
                             )
        else:
            assert result == [], "Expression {!s} and {!s} did match but were not supposed to".format(
                expression, pattern
            )

    @pytest.mark.parametrize(
        '   expression,             pattern,        expected_matches',
        [
            (f_a(a),                 f_a(x_),         [{'x': a}]),
            (f_a(a, b),              f_a(x_),         [{'x': f_a(a, b)}]),
            (f_a(a, b),              f_a(a, x_),      [{'x': b}]),
            (f_a(a, b, c),           f_a(a, x_),      [{'x': f_a(b, c)}]),
            (f_a(a, b, c),           f_a(x_, c),      [{'x': f_a(a, b)}]),
            (f_a(a, b, c),           f_a(x_),         [{'x': f_a(a, b, c)}]),
            (f_a(a, b, a, b),        f_a(x_, x_),     [{'x': f_a(a, b)}]),
            (f_a(a, b, a),           f_a(x_, b, x_),  [{'x': a}]),
            (f_a(a, a, b, a, a),     f_a(x_, b, x_),  [{'x': f_a(a, a)}]),
            (f_a(a, b, c),           f_a(x_, y_),     [{'x': a,           'y': f_a(b, c)},
                                                       {'x': f_a(a, b),   'y': c}]),
            (f_a(a),                 f_a(_),          [{}]),
            (f_a(a, b),              f_a(_),          [{}]),
          # (f_a(a, b, c),           f_a(x2_),        [{'x': (a, f_a(b, c))}]),
          # (f_a(a, b),              f_a(x2_),        [{'x': (a, b)}]),
          # (f_a(a),                 f_a(x2_),        []),
        ]
    )  # yapf: disable
    def test_associative_wildcard_dot_match(self, match, expression, pattern, expected_matches):
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

    @pytest.mark.parametrize(
        '   expression,             pattern,        expected_matches',
        [
            (f_ac(a),                f_ac(_),         [{}]),
            (f_ac(a),                f_ac(x_),        [{'x': a}]),
            (f_ac(a, b),             f_ac(_),         [{}]),
            (f_ac(a, b),             f_ac(x_),        [{'x': f_ac(a, b)}]),
        ]
    )  # yapf: disable
    def test_associative_commutative_wildcard_dot_match(self, match, expression, pattern, expected_matches):
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

    @pytest.mark.parametrize(
        '   expression,                 pattern,            expected_matches',
        [
            (a,                         x___,               [{'x': [a]}]),
            (f(a),                      f(x___),            [{'x': [a]}]),
            (f(),                       f(x___),            [{'x': []}]),
            (f(a),                      x___,               [{'x': [f(a)]}]),
            (f2(a),                     f(x___),            []),
            (f(a, b),                   f(x___),            [{'x': [a, b]}]),
            (f(a, b),                   f(x___, b),         [{'x': [a]}]),
            (f(a, b),                   f(x___, a),         []),
            (f(a, b),                   f(a, x___),         [{'x': [b]}]),
            (f(a, b),                   f(x___, x___),      []),
            (f(a, a),                   f(x___, x___),      [{'x': [a]}]),
            (f(a, b),                   f(x___, y___),      [{'x': [],                'y': [a, b]},
                                                             {'x': [a],               'y': [b]},
                                                             {'x': [a, b],            'y': []}]),
            (f(a),                      f(x___, y___),      [{'x': [],                'y': [a]},
                                                             {'x': [a],               'y': []}]),
            (f(a, b, c),                f(x___, y___),      [{'x': [],                'y': [a, b, c]},
                                                             {'x': [a],               'y': [b, c]},
                                                             {'x': [a, b],            'y': [c]},
                                                             {'x': [a, b, c],         'y': []}]),
            (f(a, f2(b)),               f(x___, y___),      [{'x': [],                'y': [a, f2(b)]},
                                                             {'x': [a],               'y': [f2(b)]},
                                                             {'x': [a, f2(b)],        'y': []}]),
            (f(a, f2(b)),               f(x___, f2(y___)),  [{'x': [a],               'y': [b]}]),
            (f(a, f2(b)),               f(x___, f2(x___)),  []),
            (f(a, f2(a)),               f(x___, f2(x___)),  [{'x': [a]}]),
            (f(f2(a), f2(b)),           f(x___, x___),      []),
            (f(f2(a), f2(b)),           f(x___, y___),      [{'x': [f2(a), f2(b)],    'y': []},
                                                             {'x': [f2(a)],           'y': [f2(b)]},
                                                             {'x': [],                'y': [f2(a), f2(b)]}]),
            (f(f2(a), a),               f(x___, x___),      []),
            (f(f2(a), a),               f(f2(x___), x___),  [{'x': [a]}]),
            (f(f(a, b)),                f(x___, y___),      [{'x': [f(a, b)],         'y': []},
                                                             {'x': [],                'y': [f(a, b)]}]),
            (f(f(a, b)),                f(x___),            [{'x': [f(a, b)]}]),
            (f2(a, b),                  f(x___, y___),      []),
            (f(a, a, a),                f(x___, b, y___),   []),
            (f(a, a, a),                f(x___, a, y___),   [{'x': [],                'y': [a, a]},
                                                             {'x': [a],               'y': [a]},
                                                             {'x': [a, a],            'y': []}]),
            (f(a),                      f(x___, a, y___),   [{'x': [],                'y': []}]),
            (f(a, a),                   f(x___, a, y___),   [{'x': [a],               'y': []},
                                                             {'x': [],                'y': [a]}]),
            (f(a, b, a),                f(x___, a, y___),   [{'x': [],                'y': [b, a]},
                                                             {'x': [a, b],            'y': []}]),
            (f(a, b, a, b),             f(x___, x___),      [{'x': [a, b]}]),
            (f(a, b, a, a),             f(x___, x___),      []),
            (f(a, b, a),                f(x___, b, x___),   [{'x': [a]}]),
            (f(a, b, a, a),             f(x___, b, x___),   []),
            (f(a, a, b, a),             f(x___, b, x___),   []),
            (f(a, b, a, b, a, b, a),    f(x___, b, x___),   [{'x': [a, b, a]}]),
            (f(a, b, a, b),             f(x___, b, y___),   [{'x': [a, b, a],         'y': []},
                                                             {'x': [a],               'y': [a, b]}]),
        ]
    )  # yapf: disable
    def test_wildcard_star_match(self, match, expression, pattern, expected_matches):
        expression = expression
        pattern = Pattern(pattern)
        result = list(match(expression, pattern))
        for expected_match in expected_matches:
            _convert_match_list_to_tuple(expected_match)
            assert expected_match in result, "Expression {!s} and {!s} did not yield the match {!s} but were supposed to".format(
                expression, pattern, expected_match
            )
        for result_match in result:
            assert result_match in expected_matches, "Expression {!s} and {!s} yielded the unexpected match {!s}".format(
                expression, pattern, result_match
            )

    @pytest.mark.parametrize(
        '   expression,                 pattern,            expected_matches',
        [
            (a,                         x__,                [{'x': [a]}]),
            (f(a),                      f(x__),             [{'x': [a]}]),
            (f(),                       f(x__),             []),
            (f(a),                      x__,                [{'x': [f(a)]}]),
            (f2(a),                     f(x__),             []),
            (f(a, b),                   f(x__),             [{'x': [a, b]}]),
            (f(a, b),                   f(x__, b),          [{'x': [a]}]),
            (f(a, b),                   f(x__, a),          []),
            (f(a, b),                   f(a, x__),          [{'x': [b]}]),
            (f(a, b),                   f(x__, x__),        []),
            (f(a, a),                   f(x__, x__),        [{'x': [a]}]),
            (f(a, b),                   f(x__, y__),        [{'x': [a],          'y': [b]}]),
            (f(a),                      f(x__, y__),        []),
            (f(a, b, c),                f(x__, y__),        [{'x': [a],          'y': [b, c]},
                                                             {'x': [a, b],       'y': [c]}]),
            (f(a, f2(b)),               f(x__, y__),        [{'x': [a],          'y': [f2(b)]}]),
            (f(a, f2(b)),               f(x__, f2(y__)),    [{'x': [a],          'y': [b]}]),
            (f(a, f2(b)),               f(x__, f2(x__)),    []),
            (f(a, f2(a)),               f(x__, f2(x__)),    [{'x': [a]}]),
            (f(f2(a), f2(b)),           f(x__, x__),        []),
            (f(f2(a), f2(b)),           f(x__, y__),        [{'x': [f2(a)],      'y': [f2(b)]}]),
            (f(f2(a), a),               f(x__, x__),        []),
            (f(f2(a), a),               f(f2(x__), x__),    [{'x': [a]}]),
            (f(f(a, b)),                f(x__, y__),        []),
            (f(f(a, b)),                f(x__),             [{'x': [f(a, b)]}]),
            (f2(a, b),                  f(x__, y__),        []),
            (f(a, a, a),                f(x__, b, y__),     []),
            (f(a, a, a),                f(x__, a, y__),     [{'x': [a],          'y': [a]}]),
            (f(a),                      f(x__, a, y__),     []),
            (f(a, a),                   f(x__, a, y__),     []),
            (f(a, b, a),                f(x__, a, y__),     []),
            (f(a, b, a, b),             f(x__, x__),        [{'x': [a, b]}]),
            (f(a, b, a, a),             f(x__, x__),        []),
            (f(a, b, a),                f(x__, b, x__),     [{'x': [a]}]),
            (f(a, b, a, a),             f(x__, b, x__),     []),
            (f(a, a, b, a),             f(x__, b, x__),     []),
            (f(a, b, a, b, a, b, a),    f(x__, b, x__),     [{'x': [a, b, a]}]),
            (f(a, b, a, b),             f(x__, b, y__),     [{'x': [a],          'y': [a, b]}]),
        ]
    )  # yapf: disable
    def test_wildcard_plus_match(self, match, expression, pattern, expected_matches):
        expression = expression
        pattern = Pattern(pattern)
        result = list(match(expression, pattern))
        for expected_match in expected_matches:
            _convert_match_list_to_tuple(expected_match)
            assert expected_match in result, "Expression {!s} and {!s} did not yield the match {!s} but were supposed to".format(
                expression, pattern, expected_match
            )
        for result_match in result:
            assert result_match in expected_matches, "Expression {!s} and {!s} yielded the unexpected match {!s}".format(
                expression, pattern, result_match
            )

    @pytest.mark.parametrize(
        '   expression,                 pattern,            expected_matches',
        [
            (f(a, b),                   f(x__, y___),       [{'x': [a, b],      'y': []},
                                                             {'x': [a],         'y': [b]}]),
            (f(a, b),                   f(x___, y__),       [{'x': [a],         'y': [b]},
                                                             {'x': [],          'y': [a, b]}]),
            (f(a, b, c),                f(x_, y__),         [{'x': a,           'y': [b, c]}]),
            (f(a, b, c),                f(x__, y_),         [{'x': [a, b],      'y': c}]),
            (f(a, b, c),                f(x_, y___),        [{'x': a,           'y': [b, c]}]),
            (f(a, b, c),                f(x___, y_),        [{'x': [a, b],      'y': c}]),
            (f(a, b, c),                f(x___, y_, z___),  [{'x': [a, b],      'y': c,        'z': []},
                                                             {'x': [a],         'y': b,        'z': [c]},
                                                             {'x': [],          'y': a,        'z': [b, c]}]),
            (f(a, b, c),                f(x__, y_, z___),   [{'x': [a, b],      'y': c,        'z': []},
                                                             {'x': [a],         'y': b,        'z': [c]}]),
            (f(a, b, c),                f(x___, y_, z__),   [{'x': [a],         'y': b,        'z': [c]},
                                                             {'x': [],          'y': a,        'z': [b, c]}]),
            (f(a, a),                   f(x_, x__),         []),
            (f(a, a),                   f(x__, x_),         []),
            (f_c(a, a),                 f_c(x__, x_),       []),
            (f(a, f_c(a)),              f(x__, f_c(x_)),    []),
            (f(a, f_c(a)),              f(x_, f_c(x__)),    []),
            (f(f_c(a), a),              f(f_c(x_), x__),    []),
            (f(f_c(a), a),              f(f_c(x__), x_),    []),
        ]
    )  # yapf: disable
    def test_wildcard_mixed_match(self, match, expression, pattern, expected_matches):
        expression = expression
        pattern = Pattern(pattern)
        result = list(match(expression, pattern))
        for expected_match in expected_matches:
            _convert_match_list_to_tuple(expected_match)
            assert expected_match in result, "Expression {!s} and {!s} did not yield the match {!s} but were supposed to".format(
                expression, pattern, expected_match
            )
        for result_match in result:
            assert result_match in expected_matches, "Expression {!s} and {!s} yielded the unexpected match {!s}".format(
                expression, pattern, result_match
            )

    @pytest.mark.parametrize(
        '   expression,             pattern,        expected_matches',
        [
            (a,                     s_,             [{'s': a}]),
            (s,                     s_,             [{'s': s}]),
            (f(a),                  s_,             []),
            (a,                     ss_,            []),
            (f(a),                  ss_,            []),
            (s,                     ss_,            [{'ss': s}]),
            (f_a(a),                f_a(s_),        [{'s': a}]),
            (f_a(f(a)),             f_a(s_),        []),
            (f_c(a),                f_c(ss_),       []),
            (f_c(s),                f_c(ss_),       [{'ss': s}]),
            (f_c(a, s),             f_c(ss_, ___),  [{'ss': s}]),
            (f_c(a, s),             f_c(s_, ___),   [{'s': s}, {'s': a}]),
            (f_ac(a),               f_ac(ss_),      []),
            (f_ac(s),               f_ac(ss_),      [{'ss': s}]),
            (f_ac(a, s),            f_ac(ss_, ___), [{'ss': s}]),
            (f_ac(a, s),            f_ac(s_, ___),  [{'s': s}, {'s': a}]),
            (f_ac(a, s),            f_ac(s_),       []),
        ]
    )  # yapf: disable
    def test_wildcard_symbol_match(self, match, expression, pattern, expected_matches):
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

    @pytest.mark.parametrize(
        '   expression,             pattern,                        expected_matches',
        [
            (f_c(a, b, a, a, b),     f_c(x_, x_, x_, y_, y_),         [{'x': a, 'y': b}]),
            (f_c(a, b, a, b),        f_c(x_, x_, y_, y_),             [{'x': a, 'y': b},
                                                                       {'x': b, 'y': a}]),
            (f_c(a, b, a, b),        f_c(x_, x_, x_, y_, y_),         []),
            (f_c(a, b, a, b),        f_c(x_, y_, y_),                 []),
            (f_c(a, b, a, b),        f_c(x_, a, y_, y_),              [{'x': a, 'y': b}]),
            (f_c(a, b, b, b),        f_c(x_, b, y_, y_),              [{'x': a, 'y': b}]),
            (f_c(a, b, a, a, b),     f_c(_, _, _, y_, y_),            [{'y': a},
                                                                       {'y': b}]),
            (f_c(a, b, a, b),        f_c(_, _, y_, y_),               [{'y': b},
                                                                       {'y': a}]),
            (f_c(a, b, a, b),        f_c(_, _, _, y_, y_),            []),
            (f_c(a, b, a, b),        f_c(_, y_, y_),                  []),
            (f_c(a, b, b, b),        f_c(_, _, y_, y_),               [{'y': b}]),
            (f_c(a, b, a, b),        f_c(_, a, y_, y_),               [{'y': b}]),
            (f_c(a, b, a, b),        f_c(_, b, y_, y_),               [{'y': a}]),
            (f_c(a, b, b, b),        f_c(_, b, y_, y_),               [{'y': b}]),
          # (f_c(a, b, a, a),        f_c(x2_, _, _),                  [{'x': Multiset([a, b])},
          #                                                            {'x': Multiset([a, a])}]),
          # (f_c(a, b, b, a),        f_c(x2_, _, _),                  [{'x': Multiset([a, b])},
          #                                                            {'x': Multiset([a, a])},
          #                                                            {'x': Multiset([b, b])}]),
        ]
    )  # yapf: disable
    def test_commutative_multiple_fixed_vars(self, match, expression, pattern, expected_matches):
        expression = expression
        pattern = Pattern(pattern)
        result = list(match(expression, pattern))
        assert len(result) == len(expected_matches), 'Unexpected number of matches'
        for expected_match in expected_matches:
            assert expected_match in result, "Expression {!s} and {!s} did not yield the match {!s} but were supposed to".format(
                expression, pattern, expected_match
            )
        for result_match in result:
            assert result_match in expected_matches, "Expression {!s} and {!s} yielded the unexpected match {!s}".format(
                expression, pattern, result_match
            )

    @pytest.mark.parametrize(
        '   expression,             pattern,                 expected_matches',
        [
            (f_c(a, b),              f_c(x___, y___),          [{'x': Multiset([a]),       'y': Multiset([b])},
                                                                {'x': Multiset([b]),       'y': Multiset([a])},
                                                                {'x': Multiset([a, b]),    'y': Multiset()},
                                                                {'x': Multiset(),          'y': Multiset([a, b])}]),
            (f_c(a, b),              f_c(x___, x___),          []),
            (f_c(a, a),              f_c(x___, x___),          [{'x': Multiset([a])}]),
            (f_c(a, a),              f_c(x___, x___, y___),    [{'x': Multiset([a]),       'y': Multiset()},
                                                                {'x': Multiset(),          'y': Multiset([a, a])}]),
            (f(a, b, f_c(a, b)),     f(x___, f_c(x___)),       [{'x': (a, b)}]),
            (f(b, a, f_c(a, b)),     f(x___, f_c(x___)),       [{'x': (b, a)}]),
            (f(a, a, f_c(a, b)),     f(x___, f_c(x___)),       []),
            (f(a, b, f_c(a, a)),     f(x___, f_c(x___)),       []),
            (f_c(a),                 f_c(__, __),              []),
            (f_c(a, a),              f_c(__, __),              [{}]),
            (f_c(a, b),              f_c(__, __),              [{}]),
        ]
    )  # yapf: disable
    def test_commutative_multiple_sequence_vars(self, match, expression, pattern, expected_matches):
        expression = expression
        pattern = Pattern(pattern)
        result = list(match(expression, pattern))
        assert len(result) == len(expected_matches), 'Unexpected number of matches'
        for expected_match in expected_matches:
            assert expected_match in result, "Expression {!s} and {!s} did not yield the match {!s} but were supposed to".format(
                expression, pattern, expected_match
            )
        for result_match in result:
            assert result_match in expected_matches, "Expression {!s} and {!s} yielded the unexpected match {!s}".format(
                expression, pattern, result_match
            )

    @pytest.mark.parametrize(
        '   expression,                 pattern,                is_match',
        [
            (f(f(a), f_c(a)),           f(f(x_), f_c(x_)),      True),
            (f(f(a, b), f_c(a, b)),     f(f(x__), f_c(x__)),    True),
            (f(f(a, b), f_c(b, a)),     f(f(x__), f_c(x__)),    True),
            (f(f(b, a), f_c(b, a)),     f(f(x__), f_c(x__)),    True),
            (f(f(b, a), f_c(a, b)),     f(f(x__), f_c(x__)),    True),
            (f(f_c(a, b), f(a, b)),     f(f_c(x__), f(x__)),    True),
            (f(f_c(b, a), f(a, b)),     f(f_c(x__), f(x__)),    True),
            (f(f_c(b, a), f(b, a)),     f(f_c(x__), f(x__)),    True),
            (f(f_c(a, b), f(b, a)),     f(f_c(x__), f(x__)),    True),
            (f(f_c(a, b), f_c(a, b)),   f(f_c(x__), f_c(x__)),  True),
        ]
    )  # yapf: disable
    def test_mixed_commutative_vars(self, match, expression, pattern, is_match):
        expression = expression
        pattern = Pattern(pattern)
        result = list(match(expression, pattern))
        if is_match:
            assert len(result) > 0
        else:
            assert len(result) == 0

    @pytest.mark.parametrize(
        '   expression,                         pattern,                    is_match',
        [
            (f(f(a), f_ac(a)),                   f(f(x_), f_ac(x_)),          True),
            (f(f_ac(a), f(a)),                   f(f_ac(x_), f(x_)),          True),
            (f(f(a, b), f_ac(a, b)),             f(f(x_), f_ac(x_)),          False),
            (f(f_ac(a, b), f(a, b)),             f(f_ac(x_), f(x_)),          False),
            (f(f_c(a), f_ac(a)),                 f(f_c(x_), f_ac(x_)),        True),
            (f(f_ac(a), f_c(a)),                 f(f_ac(x_), f_c(x_)),        True),
            (f(f_c(a, b), f_ac(a, b)),           f(f_c(x_), f_ac(x_)),        False),
            (f(f_ac(a, b), f_c(a, b)),           f(f_ac(x_), f_c(x_)),        False),
            (f(f_a(a), f_ac(a)),                 f(f_a(x_), f_ac(x_)),        True),
            (f(f_ac(a), f_a(a)),                 f(f_ac(x_), f_a(x_)),        True),
            (f(f_a(a, b), f_ac(a, b)),           f(f_a(x_), f_ac(x_)),        False),
            (f(f_ac(a, b), f_a(a, b)),           f(f_ac(x_), f_a(x_)),        False),
            (f(f(f_ac(a, b)), f_ac(a, b)),       f(f(x_), f_ac(x_)),          True),
            (f(f_ac(a, b), f(f_ac(a, b))),       f(f_ac(x_), f(x_)),          True),
            (f(f_a(a, b), f_ac(f_a(a, b))),      f(f_a(x_), f_ac(x_)),        True),
            (f(f_ac(f_a(a, b)), f_a(a, b)),      f(f_ac(x_), f_a(x_)),        True),
            (f(f_a(a, b), f_c(a, b)),            f(f_a(x_), f_c(x_, ___)),    False),
            (f(f_a(a), f_c(a, a)),               f(f_a(x_), f_c(x_, x_)),     True),
            (f(f_ac(a, b), f_ac(a, a, b, b)),    f(f_ac(x_), f_ac(x_, x_)),   True),
        ]
    )  # yapf: disable
    def test_mixed_associative_commutative_vars(self, match, expression, pattern, is_match):
        expression = expression
        pattern = Pattern(pattern)
        result = list(match(expression, pattern))
        if is_match:
            assert len(result) > 0
        else:
            assert len(result) == 0

    @pytest.mark.parametrize(
        'expression,    pattern,        constraint_values,  match_count',
        [
            (a,         _,              [False],            0),
            (a,         _,              [True],             1),
            (f(a, b),   f(_, _),        [False, True],      0),
            (f(a, b),   f(_, _),        [False, False],     0),
            (f(a, b),   f(_, _),        [True,  False],     0),
            (f(a, b),   f(_, _),        [True,  True],      1),
            (a,         x_,             [True],             1),
            (a,         x_,             [False],            0),
            (f(a, a),   f(x_, x_),      [False, False],     0),
            (f(a, a),   f(x_, x_),      [True,  False],     0),
            (f(a, a),   f(x_, x_),      [False, True],      0),
            (f(a, a),   f(x_, x_),      [True,  True],      1),
            (f(a),      f(a),           [False],            0),
            (f(a),      f(a),           [True],             1),
        ]
    )  # yapf: disable
    def test_global_constraint_syntactic_match(
            self, match_syntactic, expression, pattern, constraint_values, match_count
    ):
        constraints = [MockConstraint(v) for v in constraint_values]
        pattern = Pattern(pattern, *constraints)
        expression = expression
        result = list(match_syntactic(expression, pattern))
        assert len(result) == match_count, "Wrong number of matched for {!r} and {!r}".format(expression, pattern)

    @pytest.mark.parametrize(
        'expression,    pattern,        constraint_values,  match_count',
        [
            (a,         x_,             [True],             1),
            (a,         x_,             [False],            0),
            (f(a, a),   f(x_, x_),      [False, False],     0),
            (f(a, a),   f(x_, x_),      [True,  False],     0),
            (f(a, a),   f(x_, x_),      [False, True],      0),
            (f(a, a),   f(x_, x_),      [True,  True],      1),
        ]
    )  # yapf: disable
    def test_local_constraint_syntactic_match(
            self, match_syntactic, expression, pattern, constraint_values, match_count
    ):
        constraints = [MockConstraint(v, 'x') for v in constraint_values]
        pattern = Pattern(pattern, *constraints)
        expression = expression
        result = list(match_syntactic(expression, pattern))
        assert len(result) == match_count, "Wrong number of matched for {!r} and {!r}".format(expression, pattern)

    @pytest.mark.parametrize(
        'expression,        pattern,            constraint_values,  match_count',
        [
            (f(a, b),       f(___, ___),        [False, False],     0),
            (f(a, b),       f(___, ___),        [False, True],      0),
            (f(a, b),       f(___, ___),        [True,  False],     0),
            (f(a, b),       f(___, ___),        [True,  True],      3),
            (f_c(a, a),     f_c(x_, x_),        [False, False],     0),
            (f_c(a, a),     f_c(x_, x_),        [True,  False],     0),
            (f_c(a, a),     f_c(x_, x_),        [False, True],      0),
            (f_c(a, a),     f_c(x_, x_),        [True,  True],      1),
            (f(a, f_c(a)),  f(x_, f_c(x_)),     [False, False],     0),
            (f(a, f_c(a)),  f(x_, f_c(x_)),     [True, False],      0),
            (f(a, f_c(a)),  f(x_, f_c(x_)),     [False, True],      0),
            (f(a, f_c(a)),  f(x_, f_c(x_)),     [True, True],       1),
            (f_c(a, a),     f_c(x_, x_),        [True,  True],      1),
            (f_c(a, f(a)),  f_c(x_, f(x_)),     [False, False],     0),
            (f_c(a, f(a)),  f_c(x_, f(x_)),     [True, False],      0),
            (f_c(a, f(a)),  f_c(x_, f(x_)),     [False, True],      0),
            (f_c(a, f(a)),  f_c(x_, f(x_)),     [True, True],       1),
            (f_c(a, f(a)),  f_c(x_, f(x_)),     [False, False],     0),
            (f_c(a, f(a)),  f_c(x_, f(x_)),     [True, False],      0),
            (f_c(a, f(a)),  f_c(x_, f(x_)),     [False, True],      0),
            (f_c(a, f(a)),  f_c(x_, f(x_)),     [True, True],       1),
            (f_c(a, a),     f_c(x___),          [False],            0),
            (f_c(a, a),     f_c(x___),          [True],             1),
        ]
    )  # yapf: disable
    def test_global_constraint_non_syntactic_match(self, match, expression, pattern, constraint_values, match_count):
        constraints = [MockConstraint(v) for v in constraint_values]
        pattern = Pattern(pattern, *constraints)
        expression = expression
        result = list(match(expression, pattern))
        assert len(result) == match_count, "Wrong number of matched for {!r} and {!r}".format(expression, pattern)

    @pytest.mark.parametrize(
        'expression,        pattern,            constraint_values,  match_count',
        [
            (f_c(a, a),     f_c(x_, x_),        [False, False],     0),
            (f_c(a, a),     f_c(x_, x_),        [True,  False],     0),
            (f_c(a, a),     f_c(x_, x_),        [False, True],      0),
            (f_c(a, a),     f_c(x_, x_),        [True,  True],      1),
            (f(a, f_c(a)),  f(x_, f_c(x_)),     [False, False],     0),
            (f(a, f_c(a)),  f(x_, f_c(x_)),     [True, False],      0),
            (f(a, f_c(a)),  f(x_, f_c(x_)),     [False, True],      0),
            (f(a, f_c(a)),  f(x_, f_c(x_)),     [True, True],       1),
            (f_c(a, a),     f_c(x_, x_),        [True,  True],      1),
            (f_c(a, f(a)),  f_c(x_, f(x_)),     [False, False],     0),
            (f_c(a, f(a)),  f_c(x_, f(x_)),     [True, False],      0),
            (f_c(a, f(a)),  f_c(x_, f(x_)),     [False, True],      0),
            (f_c(a, f(a)),  f_c(x_, f(x_)),     [True, True],       1),
            (f_c(a, f(a)),  f_c(x_, f(x_)),     [False, False],     0),
            (f_c(a, f(a)),  f_c(x_, f(x_)),     [True, False],      0),
            (f_c(a, f(a)),  f_c(x_, f(x_)),     [False, True],      0),
            (f_c(a, f(a)),  f_c(x_, f(x_)),     [True, True],       1),
            (f_c(a, a),     f_c(x___),          [False],            0),
            (f_c(a, a),     f_c(x___),          [True],             1),
        ]
    )  # yapf: disable
    def test_local_constraint_non_syntactic_match(self, match, expression, pattern, constraint_values, match_count):
        constraints = [MockConstraint(v, 'x') for v in constraint_values]
        pattern = Pattern(pattern, *constraints)
        expression = expression
        result = list(match(expression, pattern))
        assert len(result) == match_count, "Wrong number of matched for {!r} and {!r}".format(expression, pattern)

    def test_constraint_call_values(self, match):
        constraint1 = MockConstraint(True, 'x')
        constraint2 = MockConstraint(True, 'y')
        constraint3 = MockConstraint(True, 'x', 'y')
        constraint4 = MockConstraint(True)
        expression = f(a, b)
        pattern = Pattern(f(___, x_, y_), constraint1, constraint2, constraint3, constraint4)
        result = list(match(expression, pattern))

        assert result == [{'x': a, 'y': b}]
        constraint1.assert_called_with({'x': a})
        constraint2.assert_called_with({'x': a, 'y': b})
        constraint3.assert_called_with({'x': a, 'y': b})
        constraint4.assert_called_with({'x': a, 'y': b})

    def test_selective_constraint(self, match):
        c = CustomConstraint(lambda x: len(str(x)) > 1)

        pattern = Pattern(f(___, x_, ___), c)
        subject = f(a, Symbol('aa'), b, Symbol('bb'))

        result = list(match(subject, pattern))

        assert len(result) == 2
        assert {'x': Symbol('aa')} in result
        assert {'x': Symbol('bb')} in result


def func_wrap_strategy(args, func):
    min_size = func.arity[0]
    max_size = func.arity[1] and func.arity[0] or 4
    return st.lists(args, min_size=min_size, max_size=max_size).map(lambda a: func(*a))


def expression_recurse_strategy(args):
    return func_wrap_strategy(args, f) | func_wrap_strategy(args, f2)


expression_base_strategy = st.sampled_from([e for e in [a, b, c]])
pattern_base_strategy = st.sampled_from([e for e in [a, b, c, x_, y_, x__, y__, x___, y___]])
expression_strategy = st.recursive(expression_base_strategy, expression_recurse_strategy, max_leaves=10)
pattern_strategy = st.recursive(pattern_base_strategy, expression_recurse_strategy, max_leaves=10)


@pytest.mark.skip("Takes too long on average")
@given(expression_strategy, pattern_strategy)
def test_randomized_match(match, expression, pattern):
    assume(not pattern.is_constant)

    expr_symbols = expression.symbols
    pattern_symbols = pattern.symbols

    # Pattern must not be just a single variable
    assume(len(pattern_symbols) > 0)

    # Pattern cannot contain symbols which are not contained in the expression
    assume(pattern_symbols <= expr_symbols)

    results = list(match(expression, pattern))

    # Exclude non-matching pairs
    assume(len(results) > 0)
    for result in results:
        reverse = substitute(pattern.expression, result)
        if isinstance(reverse, list) and len(reverse) == 1:
            reverse = reverse[0]
        assert expression == reverse
