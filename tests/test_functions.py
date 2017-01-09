# -*- coding: utf-8 -*-
import hypothesis.strategies as st
from hypothesis import assume, given
import pytest
from multiset import Multiset

from matchpy.expressions import (Arity, Constraint, Operation, Symbol,
                                        Variable, Wildcard, freeze)
from matchpy.functions import (ReplacementRule, replace, replace_all,
                                      substitute)
from matchpy.matching.one_to_one import match_anywhere

from .utils import MockConstraint


class SpecialSymbol(Symbol):
    pass

f = Operation.new('f', Arity.variadic)
f2 = Operation.new('f2', Arity.variadic)
fc = Operation.new('fc', Arity.variadic, commutative=True)
fc2 = Operation.new('fc2', Arity.variadic, commutative=True)
fa = Operation.new('fa', Arity.variadic, associative=True)
fa2 = Operation.new('fa2', Arity.variadic, associative=True)
fac1 = Operation.new('fac1', Arity.variadic, associative=True, commutative=True)
fac2 = Operation.new('fac2', Arity.variadic, associative=True, commutative=True)
a = freeze(Symbol('a'))
b = freeze(Symbol('b'))
c = freeze(Symbol('c'))
s = SpecialSymbol('s')
_ = Wildcard.dot()
x_ = Variable.dot('x')
x2_ = Variable.fixed('x', 2)
y_ = Variable.dot('y')
z_ = Variable.dot('z')
s_ = Variable.symbol('s')
ss_ = Variable.symbol('ss', SpecialSymbol)
__ = Wildcard.plus()
x__ = Variable.plus('x')
y__ = Variable.plus('y')
z__ = Variable.plus('z')
___ = Wildcard.star()
x___ = Variable.star('x')
y___ = Variable.star('y')
z___ = Variable.star('z')


mock_constraint_false = MockConstraint(False)
mock_constraint_true = MockConstraint(True)




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
    )
    def test_constant_match(self, match, expression, pattern, is_match):
        expression = freeze(expression)
        pattern = freeze(pattern)
        result = list(match(expression, pattern))
        if is_match:
            assert result == [dict()], "Expression {!s} and {!s} did not match but were supposed to".format(expression, pattern)
        else:
            assert result == [], "Expression {!s} and {!s} did match but were not supposed to".format(expression, pattern)

    @pytest.mark.parametrize(
        '   expression,         pattern,            is_match',
        [
            (fc(),              fc(),               True),
            (fc(a),             fc(),               False),
            (fc(),              fc(a),              False),
            (fc(a, b),          fc(a, b),           True),
            (fc(a, b),          fc(a, b),           True),
            (fc(b, a),          fc(a, b),           True),
            (fc(b, a, c),       fc(a, b, c),        True),
            (fc(c, a, b),       fc(a, b, c),        True),
            (fc(b, a, c),       fc(c, b, a),        True),
            (fc(b, a, a),       fc(a, a, b),        True),
            (fc(a, b, a),       fc(a, a, b),        True),
            (fc(b, b, a),       fc(a, a, b),        False),
            (fc(c, a, f2(b)),   fc(a, f2(b), c),    True),
            (fc(c, a, f2(b)),   fc(f2(a), b, c),    False),
            (f2(c, fc(a, b)),   f2(c, fc(b, a)),    True),
            (f2(c, fc(a, b)),   f2(fc(a, b), c),    False),
            (fc(c, fc(a, b)),   fc(fc(a, b), c),    True),
            (fc(c, fc(b, a)),   fc(fc(a, b), c),    True),
            (fc(c, fc(b, b)),   fc(fc(a, b), c),    False),
            (fc(a, fc(b, a)),   fc(fc(a, b), c),    False),
        ]
    )
    def test_commutative_match(self, match, expression, pattern, is_match):
        expression = freeze(expression)
        pattern = freeze(pattern)
        result = list(match(expression, pattern))
        if is_match:
            assert result == [dict()], "Expression {!s} and {!s} did not match but were supposed to".format(expression, pattern)
        else:
            assert result == [], "Expression {!s} and {!s} did match but were not supposed to".format(expression, pattern)

    @pytest.mark.parametrize(
        '   expression,         pattern,                    match_count',
        [
            (fc(f(a)),          fc(f(x_)),                  1),
            (fc(f(a), f(b)),    fc(f(x_), f(y_)),           2),
            (fc(f(a), f(b)),    fc(f(x_), f(x_)),           0),
            (fc(f(a), f(a)),    fc(f(x_), f(x_)),           1),
            (fc(f(a), f(b)),    fc(f(x_), y_),              2),
            (fc(f(a), f(a)),    fc(f(x_), x_),              0),
            (fc(f(a), a),       fc(f(x_), x_),              1),
            (fc(f(a), a),       fc(f(x_)),                  0),
            (fc(f(a), a),       fc(f(x_), f(y_)),           0),
            (fc(f(a), f(a)),    fc(f(x_), f(y_), f(z_)),    0),
            (fc(f2(a), f2(a)),  fc(f(x_), f(y_)),           0),
            (fc2(f(a),  f(a)),  fc(f(x_), f(y_)),           0),
            (fc(fc(a, b), c),   fc(fc(x__), y__),           1),
        ]
    )
    def test_commutative_syntactic_match(self, match, expression, pattern, match_count):
        expression = freeze(expression)
        pattern = freeze(pattern)
        result = list(match(expression, pattern))
        assert len(result) == match_count, 'Wrong number of matches'

        for subst in result:
            assert substitute(pattern, subst)[0] == expression, 'Invalid match'

    @pytest.mark.parametrize(
        '   expression,         pattern,                                    expected_matches',
        [
            (fc(a),             fc(x_, constraint=mock_constraint_false),   []),
            (fc(a),             fc(x_, constraint=mock_constraint_true),    [{'x': a}]),
        ]
    )
    def test_commutative_constraint_match(self, match, expression, pattern, expected_matches):
        expression = freeze(expression)
        pattern = freeze(pattern)
        result = list(match(expression, pattern))
        for expected_match in expected_matches:
            assert expected_match in result, "Expression {!s} and {!s} did not yield the match {!s} but were supposed to".format(expression, pattern, expected_match)
        for result_match in result:
            assert result_match in expected_matches, "Expression {!s} and {!s} yielded the unexpected match {!s}".format(expression, pattern, result_match)

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
            (f(a, b, c),        f(x2_),                             None),
            (f(a, b),           f(x2_),                             {'x': (a, b)}),
            (f(a),              f(x2_),                             None),
        ])
    def test_wildcard_dot_match(self, match, expression, pattern, expected_match):
        expression = freeze(expression)
        pattern = freeze(pattern)
        result = list(match(expression, pattern))
        if expected_match is not None:
            assert result == [expected_match], "Expression {!s} and {!s} did not match as {!s} but were supposed to".format(expression, pattern, expected_match)
        else:
            assert result == [], "Expression {!s} and {!s} did match but were not supposed to".format(expression, pattern)

    @pytest.mark.parametrize(
        '   expression,             pattern,        expected_matches',
        [
            (fa(a),                 fa(x_),         [{'x': a}]),
            (fa(a, b),              fa(x_),         [{'x': fa(a, b)}]),
            (fa(a, b),              fa(a, x_),      [{'x': b}]),
            (fa(a, b, c),           fa(a, x_),      [{'x': fa(b, c)}]),
            (fa(a, b, c),           fa(x_, c),      [{'x': fa(a, b)}]),
            (fa(a, b, c),           fa(x_),         [{'x': fa(a, b, c)}]),
            (fa(a, b, a, b),        fa(x_, x_),     [{'x': fa(a, b)}]),
            (fa(a, b, a),           fa(x_, b, x_),  [{'x': a}]),
            (fa(a, a, b, a, a),     fa(x_, b, x_),  [{'x': fa(a, a)}]),
            (fa(a, b, c),           fa(x_, y_),     [{'x': a,           'y': fa(b, c)},
                                                     {'x': fa(a, b),    'y': c}]),
            (fa(a, b, c),           fa(x2_),        [{'x': (a, fa(b, c))}]),
            (fa(a, b),              fa(x2_),        [{'x': (a, b)}]),
            (fa(a),                 fa(x2_),        []),
        ]
    )
    def test_associative_wildcard_dot_match(self, match, expression, pattern, expected_matches):
        expression = freeze(expression)
        pattern = freeze(pattern)
        result = list(match(expression, pattern))
        for expected_match in expected_matches:
            assert expected_match in result, "Expression {!s} and {!s} did not yield the match {!s} but were supposed to".format(expression, pattern, expected_match)
        for result_match in result:
            assert result_match in expected_matches, "Expression {!s} and {!s} yielded the unexpected match {!s}".format(expression, pattern, result_match)

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
    )
    def test_wildcard_star_match(self, match, expression, pattern, expected_matches):
        expression = freeze(expression)
        pattern = freeze(pattern)
        result = list(match(expression, pattern))
        for expected_match in expected_matches:
            _convert_match_list_to_tuple(expected_match)
            assert expected_match in result, "Expression {!s} and {!s} did not yield the match {!s} but were supposed to".format(expression, pattern, expected_match)
        for result_match in result:
            assert result_match in expected_matches, "Expression {!s} and {!s} yielded the unexpected match {!s}".format(expression, pattern, result_match)

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
    )
    def test_wildcard_plus_match(self, match, expression, pattern, expected_matches):
        expression = freeze(expression)
        pattern = freeze(pattern)
        result = list(match(expression, pattern))
        for expected_match in expected_matches:
            _convert_match_list_to_tuple(expected_match)
            assert expected_match in result, "Expression {!s} and {!s} did not yield the match {!s} but were supposed to".format(expression, pattern, expected_match)
        for result_match in result:
            assert result_match in expected_matches, "Expression {!s} and {!s} yielded the unexpected match {!s}".format(expression, pattern, result_match)

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
        ]
    )
    def test_wildcard_mixed_match(self, match, expression, pattern, expected_matches):
        expression = freeze(expression)
        pattern = freeze(pattern)
        result = list(match(expression, pattern))
        for expected_match in expected_matches:
            _convert_match_list_to_tuple(expected_match)
            assert expected_match in result, "Expression {!s} and {!s} did not yield the match {!s} but were supposed to".format(expression, pattern, expected_match)
        for result_match in result:
            assert result_match in expected_matches, "Expression {!s} and {!s} yielded the unexpected match {!s}".format(expression, pattern, result_match)

    @pytest.mark.parametrize(
        '   expression,           pattern,      expected_matches',
        [
            (a,                   s_,           [{'s': a}]),
            (s,                   s_,           [{'s': s}]),
            (f(a),                s_,           []),
            (a,                   ss_,          []),
            (f(a),                ss_,          []),
            (s,                   ss_,          [{'ss': s}]),
            (fc(a),               fc(ss_),      []),
            (fc(s),               fc(ss_),      [{'ss': s}]),
            (fc(a, s),            fc(ss_, ___), [{'ss': s}]),
            (fc(a, s),            fc(s_, ___),  [{'s': s}, {'s': a}]),
        ]
    )
    def test_wildcard_symbol_match(self, match, expression, pattern, expected_matches):
        expression = freeze(expression)
        pattern = freeze(pattern)
        result = list(match(expression, pattern))
        for expected_match in expected_matches:
            assert expected_match in result, "Expression {!s} and {!s} did not yield the match {!s} but were supposed to".format(expression, pattern, expected_match)
        for result_match in result:
            assert result_match in expected_matches, "Expression {!s} and {!s} yielded the unexpected match {!s}".format(expression, pattern, result_match)

    @pytest.mark.parametrize(
        '   expression,             pattern,                        expected_matches',
        [
            (fc(a, b, a, a, b),     fc(x_, x_, x_, y_, y_),         [{'x': a, 'y': b}]),
            (fc(a, b, a, b),        fc(x_, x_, y_, y_),             [{'x': a, 'y': b},
                                                                     {'x': b, 'y': a}]),
            (fc(a, b, a, b),        fc(x_, x_, x_, y_, y_),         []),
            (fc(a, b, a, b),        fc(x_, y_, y_),                 []),
            (fc(a, b, a, b),        fc(x_, a, y_, y_),              [{'x': a, 'y': b}]),
            (fc(a, b, b, b),        fc(x_, b, y_, y_),              [{'x': a, 'y': b}]),
            (fc(a, b, a, a, b),     fc(_, _, _, y_, y_),            [{'y': a},
                                                                     {'y': b}]),
            (fc(a, b, a, b),        fc(_, _, y_, y_),               [{'y': b},
                                                                     {'y': a}]),
            (fc(a, b, a, b),        fc(_, _, _, y_, y_),            []),
            (fc(a, b, a, b),        fc(_, y_, y_),                  []),
            (fc(a, b, b, b),        fc(_, _, y_, y_),               [{'y': b}]),
            (fc(a, b, a, b),        fc(_, a, y_, y_),               [{'y': b}]),
            (fc(a, b, a, b),        fc(_, b, y_, y_),               [{'y': a}]),
            (fc(a, b, b, b),        fc(_, b, y_, y_),               [{'y': b}]),
            (fc(a, b, a, a),        fc(x2_, _, _),                  [{'x': Multiset([a, b])},
                                                                     {'x': Multiset([a, a])}]),
            (fc(a, b, b, a),        fc(x2_, _, _),                  [{'x': Multiset([a, b])},
                                                                     {'x': Multiset([a, a])},
                                                                     {'x': Multiset([b, b])}]),
        ]
    )
    def test_commutative_multiple_fixed_vars(self, match, expression, pattern, expected_matches):
        expression = freeze(expression)
        pattern = freeze(pattern)
        result = list(match(expression, pattern))
        assert len(result) == len(expected_matches), 'Unexpected number of matches'
        for expected_match in expected_matches:
            assert expected_match in result, "Expression {!s} and {!s} did not yield the match {!s} but were supposed to".format(expression, pattern, expected_match)
        for result_match in result:
            assert result_match in expected_matches, "Expression {!s} and {!s} yielded the unexpected match {!s}".format(expression, pattern, result_match)

    @pytest.mark.parametrize(
        '   expression,             pattern,                 expected_matches',
        [
            (fc(a, b),              fc(x___, y___),          [{'x': Multiset([a]),       'y': Multiset([b])},
                                                              {'x': Multiset([b]),       'y': Multiset([a])},
                                                              {'x': Multiset([a, b]),    'y': Multiset()},
                                                              {'x': Multiset(),          'y': Multiset([a, b])}]),
            (fc(a, b),              fc(x___, x___),          []),
            (fc(a, a),              fc(x___, x___),          [{'x': Multiset([a])}]),
            (fc(a, a),              fc(x___, x___, y___),    [{'x': Multiset([a]),       'y': Multiset()},
                                                              {'x': Multiset(),          'y': Multiset([a, a])}]),
            (f(a, b, fc(a, b)),     f(x___, fc(x___)),       [{'x': (a, b)}]),
            (f(b, a, fc(a, b)),     f(x___, fc(x___)),       [{'x': (b, a)}]),
            (f(a, a, fc(a, b)),     f(x___, fc(x___)),       []),
            (f(a, b, fc(a, a)),     f(x___, fc(x___)),       []),
            (fc(a),                 fc(__, __),              []),
            (fc(a, a),              fc(__, __),              [{}]),
            (fc(a, b),              fc(__, __),              [{}]),
        ]
    )
    def test_commutative_multiple_sequence_vars(self, match, expression, pattern, expected_matches):
        expression = freeze(expression)
        pattern = freeze(pattern)
        result = list(match(expression, pattern))
        assert len(result) == len(expected_matches), 'Unexpected number of matches'
        for expected_match in expected_matches:
            assert expected_match in result, "Expression {!s} and {!s} did not yield the match {!s} but were supposed to".format(expression, pattern, expected_match)
        for result_match in result:
            assert result_match in expected_matches, "Expression {!s} and {!s} yielded the unexpected match {!s}".format(expression, pattern, result_match)

    @pytest.mark.parametrize(
        '   expression,             pattern,                is_match',
        [
            (f(f(a), fc(a)),        f(f(x_), fc(x_)),       True),
            (f(f(a, b), fc(a, b)),  f(f(x__), fc(x__)),     True),
            (f(f(a, b), fc(b, a)),  f(f(x__), fc(x__)),     True),
            (f(f(b, a), fc(b, a)),  f(f(x__), fc(x__)),     True),
            (f(f(b, a), fc(a, b)),  f(f(x__), fc(x__)),     True),
            (f(fc(a, b), f(a, b)),  f(fc(x__), f(x__)),     True),
            (f(fc(b, a), f(a, b)),  f(fc(x__), f(x__)),     True),
            (f(fc(b, a), f(b, a)),  f(fc(x__), f(x__)),     True),
            (f(fc(a, b), f(b, a)),  f(fc(x__), f(x__)),     True),
        ]
    )
    def test_mixed_commutative_vars(self, match, expression, pattern, is_match):
        expression = freeze(expression)
        pattern = freeze(pattern)
        result = list(match(expression, pattern))
        if is_match:
            assert len(result) > 0
        else:
            assert len(result) == 0

    @pytest.mark.parametrize(
        'expression,    pattern_factory,                                                    constraint_values,  constraint_call_counts, match_count',
        [
            (a,         lambda c: Wildcard(1, True, c),                                     [False],            [1],                    0),
            (a,         lambda c: Wildcard(1, True, c),                                     [True],             [1],                    1),
            (f(a, b),   lambda c1, c2: f(Wildcard(1, True, c1), Wildcard(1, True, c2)),     [False, True],      [1, 0],                 0),
            (f(a, b),   lambda c1, c2: f(Wildcard(1, True, c1), Wildcard(1, True, c2)),     [False, False],     [1, 0],                 0),
            (f(a, b),   lambda c1, c2: f(Wildcard(1, True, c1), Wildcard(1, True, c2)),     [True,  False],     [1, 1],                 0),
            (f(a, b),   lambda c1, c2: f(Wildcard(1, True, c1), Wildcard(1, True, c2)),     [True,  True],      [1, 1],                 1),
            (f(a, b),   lambda c1, c2: f(Wildcard(0, False, c1), Wildcard(0, False, c2)),   [False, False],     [3, 0],                 0),
            (f(a, b),   lambda c1, c2: f(Wildcard(0, False, c1), Wildcard(0, False, c2)),   [False, True],      [3, 0],                 0),
            (f(a, b),   lambda c1, c2: f(Wildcard(0, False, c1), Wildcard(0, False, c2)),   [True,  False],     [3, 3],                 0),
            (f(a, b),   lambda c1, c2: f(Wildcard(0, False, c1), Wildcard(0, False, c2)),   [True,  True],      [3, 3],                 3),
            (a,         lambda c: Variable('x', _, c),                                      [True],             [1],                    1),
            (a,         lambda c: Variable('x', _, c),                                      [False],            [1],                    0),
            (f(a, a),   lambda c1, c2: f(Variable('x', _, c1), Variable('x', _, c2)),       [False, False],     [1, 0],                 0),
            (f(a, a),   lambda c1, c2: f(Variable('x', _, c1), Variable('x', _, c2)),       [True,  False],     [1, 1],                 0),
            (f(a, a),   lambda c1, c2: f(Variable('x', _, c1), Variable('x', _, c2)),       [False, True],      [1, 0],                 0),
            (f(a, a),   lambda c1, c2: f(Variable('x', _, c1), Variable('x', _, c2)),       [True,  True],      [1, 1],                 1),
            (f(a),      lambda c: f(a, constraint=c),                                       [False],            [1],                    0),
            (f(a),      lambda c: f(a, constraint=c),                                       [True],             [1],                    1),
        ]
    )
    def test_constraint_match(self, match, expression, pattern_factory, constraint_values, constraint_call_counts, match_count):
        constraints = [MockConstraint(v) for v in constraint_values]
        pattern = pattern_factory(*constraints)
        expression = freeze(expression)
        pattern = freeze(pattern)
        result = list(match(expression, pattern))

        assert len(result) == match_count, "Wrong number of matched for {!r} and {!r}".format(expression, pattern)
        for constraint, call_count in zip(constraints, constraint_call_counts):
            assert constraint.call_count >= call_count

    def test_constraint_call_values(self, match):
        constraint1 = MockConstraint(True)
        constraint2 = MockConstraint(True)
        constraint3 = MockConstraint(True)
        constraint4 = MockConstraint(True)
        expression = freeze(f(a, b))
        pattern = f(Wildcard(0, False, constraint1), Variable('x', _, constraint2), Variable('y', _, constraint3), constraint=constraint4)

        pattern = freeze(pattern)
        result = list(match(expression, pattern))

        assert result == [{'x': a, 'y': b}]
        constraint1.assert_called_with({})
        constraint2.assert_called_with({'x': a})
        constraint3.assert_called_with({'x': a, 'y': b})
        constraint4.assert_called_with({'x': a, 'y': b})

    def test_wildcard_internal_match(self):
        from matchpy.matching.common import _match

        matches = list(_match([a, b], x_, {}))
        assert matches == []

        matches = list(_match([], x_, {}))
        assert matches == []

        matches = list(_match([], x__, {}))
        assert matches == []


def func_wrap_strategy(args, func):
    min_size = func.arity[0]
    max_size = func.arity[1] and func.arity[0] or 4
    return st.lists(args, min_size=min_size, max_size=max_size).map(lambda a: freeze(func(*a)))


def expression_recurse_strategy(args):
    return func_wrap_strategy(args, f) | func_wrap_strategy(args, f2)

expression_base_strategy = st.sampled_from([freeze(e) for e in [a, b, c]])
pattern_base_strategy = st.sampled_from([freeze(e) for e in [a, b, c, x_, y_, x__, y__, x___, y___]])
expression_strategy = st.recursive(expression_base_strategy, expression_recurse_strategy, max_leaves=10)
pattern_strategy = st.recursive(pattern_base_strategy, expression_recurse_strategy, max_leaves=10)


@pytest.mark.skip("Takes too long on average")
@given(expression_strategy, pattern_strategy)
def test_randomized_match(match, expression, pattern):
    assume(not pattern.is_constant)

    expr_symbols = expression.symbols
    pattern_symbols = pattern.symbols

    # pattern must not be just a single variable
    assume(len(pattern_symbols) > 0)

    # Pattern cannot contain symbols which are not contained in the expression
    assume(pattern_symbols <= expr_symbols)

    results = list(match(expression, pattern))

    # exclude non-matching pairs
    assume(len(results) > 0)
    for result in results:
        reverse, _ = substitute(pattern, result)
        if isinstance(reverse, list) and len(reverse) == 1:
            reverse = reverse[0]
        assert expression == reverse


class TestSubstitute:
    @pytest.mark.parametrize(
        '   expression,                         substitution,           expected_result,    replaced',
        [
            (a,                                 {},                     a,                  False),
            (a,                                 {'x': b},               a,                  False),
            (x_,                                {'x': b},               b,                  True),
            (x_,                                {'x': [a, b]},          [a, b],             True),
            (y_,                                {'x': b},               y_,                 False),
            (f(x_),                             {'x': b},               f(b),               True),
            (f(x_),                             {'y': b},               f(x_),              False),
            (f(x_),                             {},                     f(x_),              False),
            (f(a, x_),                          {'x': b},               f(a, b),            True),
            (f(x_),                             {'x': [a, b]},          f(a, b),            True),
            (f(x_),                             {'x': []},              f(),                True),
            (f(x_, c),                          {'x': [a, b]},          f(a, b, c),         True),
            (f(x_, y_),                         {'x': a, 'y': b},       f(a, b),            True),
            (f(x_, y_),                         {'x': [a, c], 'y': b},  f(a, c, b),         True),
            (f(x_, y_),                         {'x': a, 'y': [b, c]},  f(a, b, c),         True)
        ]
    )
    def test_substitution_match(self, expression, substitution, expected_result, replaced):
        result, did_replace = substitute(expression, substitution)
        assert result == expected_result, "Substitution did not yield expected result"
        assert did_replace == replaced, "Substitution did not yield expected result"
        if not did_replace:
            assert result is expression, "When nothing is substituted, the original expression has to be returned"

    def test_error_with_nested_variables(self):
        with pytest.raises(ValueError):
            substitute(Variable('x', Variable('y', a)), {'y': [a, b]})

        with pytest.raises(ValueError):
            substitute(Variable('x', Variable('y', a)), {'y': []})


class TestReplaceTest:
    @pytest.mark.parametrize(
        '   expression,             position,   replacement,    expected_result',
        [
            (a,                     (),         b,              b),
            (f(a),                  (),         b,              b),
            (a,                     (),         f(b),           f(b)),
            (f(a),                  (),         f(b),           f(b)),
            (f(a),                  (0, ),      b,              f(b)),
            (f(a, b),               (0, ),      c,              f(c, b)),
            (f(a, b),               (1, ),      c,              f(a, c)),
            (f(a),                  (0, ),      [b, c],         f(b, c)),
            (f(a, b),               (0, ),      [b, c],         f(b, c, b)),
            (f(a, b),               (1, ),      [b, c],         f(a, b, c)),
            (f(f(a)),               (0, ),      b,              f(b)),
            (f(f(a)),               (0, 0),     b,              f(f(b))),
            (f(f(a, b)),            (0, 0),     c,              f(f(c, b))),
            (f(f(a, b)),            (0, 1),     c,              f(f(a, c))),
            (f(f(a, b), f(a, b)),   (0, 0),     c,              f(f(c, b), f(a, b))),
            (f(f(a, b), f(a, b)),   (0, 1),     c,              f(f(a, c), f(a, b))),
            (f(f(a, b), f(a, b)),   (1, 0),     c,              f(f(a, b), f(c, b))),
            (f(f(a, b), f(a, b)),   (1, 1),     c,              f(f(a, b), f(a, c))),
            (f(f(a, b), f(a, b)),   (0, ),      c,              f(c, f(a, b))),
            (f(f(a, b), f(a, b)),   (1, ),      c,              f(f(a, b), c)),
        ]
    )
    def test_substitution_match(self, expression, position, replacement, expected_result):
        result = replace(expression, position, replacement)
        assert result == expected_result, "Replacement did not yield expected result ({!r} {!r} -> {!r})".format(expression, position, replacement)
        assert result != expression, "Replacement modified the original expression"

    def test_too_big_position_error(self):
        with pytest.raises(IndexError):
            replace(a, (0, ), b)
        with pytest.raises(IndexError):
            replace(f(a), (0, 0), b)
        with pytest.raises(IndexError):
            replace(f(a), (1, ), b)
        with pytest.raises(IndexError):
            replace(f(a, b), (2, ), b)


@pytest.mark.parametrize(
    '   expression,                                             pattern,    expected_results',
    [                                                                       # Substitution      Position
        (f(a),                                                  f(x_),      [({'x': a},         ())]),
        (f(a),                                                  x_,         [({'x': f(a)},      ()),
                                                                             ({'x': a},         (0, ))]),
        (f(a, f2(b), f2(f2(c), f2(a), f2(f2(b))), f2(c), c),    f2(x_),     [({'x': b},         (1, )),
                                                                             ({'x': c},         (2, 0)),
                                                                             ({'x': a},         (2, 1)),
                                                                             ({'x': f2(b)},     (2, 2)),
                                                                             ({'x': b},         (2, 2, 0)),
                                                                             ({'x': c},         (3, ))])
    ]
)
def test_match_anywhere(expression, pattern, expected_results):
    expression = freeze(expression)
    pattern = freeze(pattern)
    results = list(match_anywhere(expression, pattern))

    assert len(results) == len(expected_results), "Invalid number of results"

    for result in expected_results:
        assert result in results, "Results differ from expected"


def test_logic_simplify():
    LAnd = Operation.new('and', Arity.variadic, 'LAnd', associative=True, one_identity=True, commutative=True)
    LOr = Operation.new('or', Arity.variadic, 'LOr', associative=True, one_identity=True, commutative=True)
    LXor = Operation.new('xor', Arity.variadic, 'LXor', associative=True, one_identity=True, commutative=True)
    LNot = Operation.new('not', Arity.unary, 'LNot')
    LImplies = Operation.new('implies', Arity.binary, 'LImplies')
    Iff = Operation.new('iff', Arity.binary, 'Iff')

    ___ = Wildcard.star()

    a1 = Symbol('a1')
    a2 = Symbol('a2')
    a3 = Symbol('a3')
    a4 = Symbol('a4')
    a5 = Symbol('a5')
    a6 = Symbol('a6')
    a7 = Symbol('a7')
    a8 = Symbol('a8')
    a9 = Symbol('a9')
    a10 = Symbol('a10')
    a11 = Symbol('a11')

    LBot = Symbol(u'⊥')
    LTop = Symbol(u'⊤')

    expression = LImplies(LAnd(Iff(Iff(LOr(a1, a2), LOr(LNot(a3), Iff(LXor(a4, a5), LNot(LNot(LNot(a6)))))),
        LNot(LAnd(LAnd(a7, a8), LNot(LXor(LXor(LOr(a9, LAnd(a10, a11)), a2), LAnd(LAnd(a11, LXor(a2, Iff(
        a5, a5))), LXor(LXor(a7, a7), Iff(a9, a4)))))))), LImplies(Iff(Iff(LOr(a1, a2), LOr(LNot(a3),
        Iff(LXor(a4, a5), LNot(LNot(LNot(a6)))))), LNot(LAnd(LAnd(a7, a8), LNot(LXor(LXor(LOr(a9, LAnd(
        a10, a11)), a2), LAnd(LAnd(a11, LXor(a2, Iff(a5, a5))), LXor(LXor(a7, a7), Iff(a9, a4)))))))),
        LNot(LAnd(LImplies(LAnd(a1, a2), LNot(LXor(LOr(LOr(LXor(LImplies(LAnd(a3, a4), LImplies(a5, a6)),
        LOr(a7, a8)), LXor(Iff(a9, a10), a11)), LXor(LXor(a2, a2), a7)), Iff(LOr(a4, a9), LXor(LNot(a6),
        a6))))), LNot(Iff(LNot(a11), LNot(a9))))))), LNot(LAnd(LImplies(LAnd(a1, a2), LNot(LXor(LOr(LOr(
        LXor(LImplies(LAnd(a3, a4), LImplies(a5, a6)), LOr(a7, a8)), LXor(Iff(a9, a10), a11)), LXor(
        LXor(a2, a2), a7)), Iff(LOr(a4, a9), LXor(LNot(a6), a6))))), LNot(Iff(LNot(a11), LNot(a9))))))

    rules = [
        # xor(x,⊥) → x
        ReplacementRule(
            LXor(x__, LBot),
            lambda x: LXor(*x)
        ),
        # xor(x, x) → ⊥
        ReplacementRule(
            LXor(x_, x_, ___),
            lambda x: LBot
        ),
        # and(x,⊤) → x
        ReplacementRule(
            LAnd(x__, LTop),
            lambda x: LAnd(*x)
        ),
        # and(x,⊥) → ⊥
        ReplacementRule(
            LAnd(___, LBot),
            lambda: LBot
        ),
        # and(x, x) → x
        ReplacementRule(
            LAnd(x_, x_, y___),
            lambda x, y: LAnd(x, *y)
        ),
        # and(x, xor(y, z)) → xor(and(x, y), and(x, z))
        ReplacementRule(
            LAnd(x_, LXor(y_, z_)),
            lambda x, y, z: LXor(LAnd(x, y), LAnd(x, z))
        ),
        # implies(x, y) → not(xor(x, and(x, y)))
        ReplacementRule(
            LImplies(x_, y_),
            lambda x, y: LNot(LXor(x, LAnd(x, y)))
        ),
        # not(x) → xor(x,⊤)
        ReplacementRule(
            LNot(x_),
            lambda x: LXor(x, LTop)
        ),
        # or(x, y) → xor(and(x, y), xor(x, y))
        ReplacementRule(
            LOr(x_, y_),
            lambda x, y: LXor(LAnd(x, y), LXor(x, y))
        ),
        # iff(x, y) → not(xor(x, y))
        ReplacementRule(
            Iff(x_, y_),
            lambda x, y: LNot(LXor(x, y))
        ),
    ]

    result = replace_all(expression, rules)

    assert result == LBot


if __name__ == '__main__':
    import matchpy.functions as tested_module
    pytest.main(['--doctest-modules', __file__, tested_module.__file__])
