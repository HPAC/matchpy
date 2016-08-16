# -*- coding: utf-8 -*-
import unittest
from ddt import ddt, data, unpack

from patternmatcher.expressions import Operation, Symbol, Variable, Arity, Wildcard
from patternmatcher.functions import match
from patternmatcher.utils import match_repr_str


f = Operation.new('f', Arity.variadic)
f2 = Operation.new('f2', Arity.variadic)
fc = Operation.new('fc', Arity.variadic, commutative=True)
fc2 = Operation.new('fc2', Arity.variadic, commutative=True)
fa = Operation.new('fa', Arity.variadic, associative=True)
fa2 = Operation.new('fa2', Arity.variadic, associative=True)
fac1 = Operation.new('fac1', Arity.variadic, associative=True, commutative=True)
fac2 = Operation.new('fac2', Arity.variadic, associative=True, commutative=True)
a = Symbol('a')
b = Symbol('b')
c = Symbol('c')
wd = Wildcard.dot()
vd = Variable.dot('vd')
vd2 = Variable.dot('vd2')
wp = Wildcard.plus()
vp = Variable.plus('vp')
vp2 = Variable.plus('vp2')
ws = Wildcard.star()
vs = Variable.star('vs')
vs2 = Variable.star('vs2')

@ddt
class MatchTest(unittest.TestCase):
    @unpack
    @data(
        (a,                 a,              True),
        (b,                 a,              False),
        (f(a),              f(a),           True),
        (f(b),              f(a),           False),
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
    )
    def test_constant_match(self, expr, pattern, is_match):
        result = list(match([expr], pattern))
        if is_match:
            self.assertEqual(result, [dict()], 'Expression %s and %s did not match but were supposed to' % (expr, pattern))
        else:
            self.assertEqual(result, [], 'Expression %s and %s did match but were not supposed to' % (expr, pattern))

    @unpack
    @data(
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
    )
    def test_commutative_match(self, expr, pattern, is_match):
        result = list(match([expr], pattern))
        if is_match:
            self.assertEqual(result, [dict()], 'Expression %s and %s did not match but were supposed to' % (expr, pattern))
        else:
            self.assertEqual(result, [], 'Expression %s and %s did match but were not supposed to' % (expr, pattern))

    @unpack
    @data(
        (a,                 vd,                 {'vd': a}),
        (b,                 vd,                 {'vd': b}),
        (f(a),              f(vd),              {'vd': a}),
        (f(b),              f(vd),              {'vd': b}),
        (f(a),              vd,                 {'vd': f(a)}),
        (f2(a),             f(vd),              None),
        (f(a, b),           f(vd),              None),
        (f(a, b),           f(vd, b),           {'vd': a}),
        (f(a, b),           f(vd, a),           None),
        (f(a, b),           f(a, vd),           {'vd': b}),
        (f(a, b),           f(vd, vd),          None),
        (f(a, a),           f(vd, vd),          {'vd': a}),
        (f(a, b),           f(vd, vd2),         {'vd': a,       'vd2': b}),
        (f(a),              f(vd, vd2),         None),
        (f(a, b, c),        f(vd, vd2),         None),
        (f(a, f2(b)),       f(vd, vd2),         {'vd': a,       'vd2': f2(b)}),
        (f(a, f2(b)),       f(vd, f2(vd2)),     {'vd': a,       'vd2': b}),
        (f(a, f2(b)),       f(vd, f2(vd)),      None),
        (f(a, f2(a)),       f(vd, f2(vd)),      {'vd': a}),
        (f(f2(a), f2(b)),   f(vd, vd),          None),
        (f(f2(a), f2(b)),   f(vd, vd2),         {'vd': f2(a),   'vd2': f2(b)}),
        (f(f2(a), a),       f(vd, vd),          None),
        (f(f2(a), a),       f(f2(vd), vd),      {'vd': a}),
        (f(f(a, b)),        f(vd, vd2),         None),
        (f(f(a, b)),        f(vd),              {'vd': f(a, b)}),
        (f2(a, b),          f(vd, vd2),         None),
        (f(f(a, b)),        f(f(vd, vd2)),      {'vd': a,       'vd2': b})
    )
    def test_wildcard_dot_match(self, expr, pattern, expected_match):
        result = list(match([expr], pattern))
        if expected_match is not None:
            self.assertEqual(result, [expected_match], 'Expression %s and %s did not match as %s but were supposed to' \
                % (expr, pattern, match_repr_str(expected_match)))
        else:
            self.assertEqual(result, [], 'Expression %s and %s did match but were not supposed to' % (expr, pattern))

    @unpack
    @data(
        (fa(a),                 fa(vd),         [{'vd': a}]),
        (fa(a, b),              fa(vd),         [{'vd': fa(a, b)}]),
        (fa(a, b),              fa(a, vd),      [{'vd': b}]),
        (fa(a, b, c),           fa(a, vd),      [{'vd': fa(b, c)}]),
        (fa(a, b, c),           fa(vd, c),      [{'vd': fa(a, b)}]),
        (fa(a, b, c),           fa(vd),         [{'vd': fa(a, b, c)}]),
        (fa(a, b, a, b),        fa(vd, vd),     [{'vd': fa(a, b)}]),
        (fa(a, b, a),           fa(vd, b, vd),  [{'vd': a}]),
        (fa(a, a, b, a, a),     fa(vd, b, vd),  [{'vd': fa(a, a)}]),
        (fa(a, b, c),           fa(vd, vd2),    [{'vd': a,          'vd2': fa(b, c)}, \
                                                 {'vd': fa(a, b),    'vd2': c}])
    )
    def test_associative_wildcard_dot_match(self, expr, pattern, expected_matches):
        result = list(match([expr], pattern))
        for expected_match in expected_matches:
            self.assertIn(expected_match, result, 'Expression %s and %s did not yield the match %s but were supposed to' % (expr, pattern, match_repr_str(expected_match)))
        for result_match in result:
            self.assertIn(result_match, expected_matches, 'Expression %s and %s yielded the unexpected match %s' % (expr, pattern, match_repr_str(result_match)))

    @unpack
    @data(
        (a,                         vs,              [{'vs': [a]}]),
        (f(a),                      f(vs),           [{'vs': [a]}]),
        (f(),                       f(vs),           [{'vs': []}]),
        (f(a),                      vs,              [{'vs': [f(a)]}]),
        (f2(a),                     f(vs),           []),
        (f(a, b),                   f(vs),           [{'vs': [a, b]}]),
        (f(a, b),                   f(vs, b),        [{'vs': [a]}]),
        (f(a, b),                   f(vs, a),        []),
        (f(a, b),                   f(a, vs),        [{'vs': [b]}]),
        (f(a, b),                   f(vs, vs),       []),
        (f(a, a),                   f(vs, vs),       [{'vs': [a]}]),
        (f(a, b),                   f(vs, vs2),      [{'vs': [],                'vs2': [a, b]},     \
                                                      {'vs': [a],               'vs2': [b]},        \
                                                      {'vs': [a, b],            'vs2': []}]),
        (f(a),                      f(vs, vs2),      [{'vs': [],                'vs2': [a]},        \
                                                      {'vs': [a],               'vs2': []}]),
        (f(a, b, c),                f(vs, vs2),      [{'vs': [],                'vs2': [a, b, c]},  \
                                                      {'vs': [a],               'vs2': [b, c]},     \
                                                      {'vs': [a, b],            'vs2': [c]},        \
                                                      {'vs': [a, b, c],         'vs2': []}]),
        (f(a, f2(b)),               f(vs, vs2),      [{'vs': [],                'vs2': [a, f2(b)]}, \
                                                      {'vs': [a],               'vs2': [f2(b)]},    \
                                                      {'vs': [a, f2(b)],        'vs2': []}]),
        (f(a, f2(b)),               f(vs, f2(vs2)),  [{'vs': [a],               'vs2': [b]}]),
        (f(a, f2(b)),               f(vs, f2(vs)),   []),
        (f(a, f2(a)),               f(vs, f2(vs)),   [{'vs': [a]}]),
        (f(f2(a), f2(b)),           f(vs, vs),       []),
        (f(f2(a), f2(b)),           f(vs, vs2),      [{'vs': [f2(a), f2(b)],    'vs2': []},         \
                                                      {'vs': [f2(a)],           'vs2': [f2(b)]},    \
                                                      {'vs': [],                'vs2': [f2(a), f2(b)]}]),
        (f(f2(a), a),               f(vs, vs),       []),
        (f(f2(a), a),               f(f2(vs), vs),   [{'vs': [a]}]),
        (f(f(a, b)),                f(vs, vs2),      [{'vs': [f(a, b)],         'vs2': []},         \
                                                      {'vs': [],                'vs2': [f(a, b)]}]),
        (f(f(a, b)),                f(vs),           [{'vs': [f(a, b)]}]),
        (f2(a, b),                  f(vs, vs2),      []),
        (f(a, a, a),                f(vs, b, vs2),   []),
        (f(a, a, a),                f(vs, a, vs2),   [{'vs': [],                'vs2': [a, a]},     \
                                                      {'vs': [a],               'vs2': [a]},        \
                                                      {'vs': [a, a],            'vs2': []}]),
        (f(a),                      f(vs, a, vs2),   [{'vs': [],                'vs2': []}]),
        (f(a, a),                   f(vs, a, vs2),   [{'vs': [a],               'vs2': []},         \
                                                     {'vs': [],                 'vs2': [a]}]),
        (f(a, b, a),                f(vs, a, vs2),   [{'vs': [],                'vs2': [b, a]},     \
                                                      {'vs': [a, b],            'vs2': []}]),
        (f(a, b, a, b),             f(vs, vs),       [{'vs': [a, b]}]),
        (f(a, b, a, a),             f(vs, vs),       []),
        (f(a, b, a),                f(vs, b, vs),    [{'vs': [a]}]),
        (f(a, b, a, a),             f(vs, b, vs),    []),
        (f(a, a, b, a),             f(vs, b, vs),    []),
        (f(a, b, a, b, a, b, a),    f(vs, b, vs),    [{'vs': [a, b, a]}]),
        (f(a, b, a, b),             f(vs, b, vs2),   [{'vs': [a, b, a],         'vs2': []},         \
                                                      {'vs': [a],               'vs2': [a, b]}]),
    )
    def test_wildcard_star_match(self, expr, pattern, expected_matches):
        result = list(match([expr], pattern))
        for expected_match in expected_matches:
            self.assertIn(expected_match, result, 'Expression %s and %s did not yield the match %s but were supposed to' % (expr, pattern, match_repr_str(expected_match)))
        for result_match in result:
            self.assertIn(result_match, expected_matches, 'Expression %s and %s yielded the unexpected match %s' % (expr, pattern, match_repr_str(result_match)))
    
    @unpack
    @data(
        (a,                         vp,                 [{'vp': [a]}]),
        (f(a),                      f(vp),              [{'vp': [a]}]),
        (f(),                       f(vp),              []),
        (f(a),                      vp,                 [{'vp': [f(a)]}]),
        (f2(a),                     f(vp),              []),
        (f(a, b),                   f(vp),              [{'vp': [a, b]}]),
        (f(a, b),                   f(vp, b),           [{'vp': [a]}]),
        (f(a, b),                   f(vp, a),           []),
        (f(a, b),                   f(a, vp),           [{'vp': [b]}]),
        (f(a, b),                   f(vp, vp),          []),
        (f(a, a),                   f(vp, vp),          [{'vp': [a]}]),
        (f(a, b),                   f(vp, vp2),         [{'vp': [a],          'vp2': [b]}]),
        (f(a),                      f(vp, vp2),         []),
        (f(a, b, c),                f(vp, vp2),         [{'vp': [a],          'vp2': [b, c]},     \
                                                         {'vp': [a, b],       'vp2': [c]}]),
        (f(a, f2(b)),               f(vp, vp2),         [{'vp': [a],          'vp2': [f2(b)]}]),
        (f(a, f2(b)),               f(vp, f2(vp2)),     [{'vp': [a],          'vp2': [b]}]),
        (f(a, f2(b)),               f(vp, f2(vp)),      []),
        (f(a, f2(a)),               f(vp, f2(vp)),      [{'vp': [a]}]),
        (f(f2(a), f2(b)),           f(vp, vp),          []),
        (f(f2(a), f2(b)),           f(vp, vp2),         [{'vp': [f2(a)],       'vp2': [f2(b)]}]),
        (f(f2(a), a),               f(vp, vp),          []),
        (f(f2(a), a),               f(f2(vp), vp),      [{'vp': [a]}]),
        (f(f(a, b)),                f(vp, vp2),         []),
        (f(f(a, b)),                f(vp),              [{'vp': [f(a, b)]}]),
        (f2(a, b),                  f(vp, vp2),         []),
        (f(a, a, a),                f(vp, b, vp2),      []),
        (f(a, a, a),                f(vp, a, vp2),      [{'vp': [a],          'vp2': [a]}]),
        (f(a),                      f(vp, a, vp2),      []),
        (f(a, a),                   f(vp, a, vp2),      []),
        (f(a, b, a),                f(vp, a, vp2),      []),
        (f(a, b, a, b),             f(vp, vp),          [{'vp': [a, b]}]),
        (f(a, b, a, a),             f(vp, vp),          []),
        (f(a, b, a),                f(vp, b, vp),       [{'vp': [a]}]),
        (f(a, b, a, a),             f(vp, b, vp),       []),
        (f(a, a, b, a),             f(vp, b, vp),       []),
        (f(a, b, a, b, a, b, a),    f(vp, b, vp),       [{'vp': [a, b, a]}]),
        (f(a, b, a, b),             f(vp, b, vp2),      [{'vp': [a],          'vp2': [a, b]}]),
    )
    def test_wildcard_plus_match(self, expr, pattern, expected_matches):
        result = list(match([expr], pattern))
        for expected_match in expected_matches:
            self.assertIn(expected_match, result, 'Expression %s and %s did not yield the match %s but were supposed to' % (expr, pattern, match_repr_str(expected_match)))
        for result_match in result:
            self.assertIn(result_match, expected_matches, 'Expression %s and %s yielded the unexpected match %s' % (expr, pattern, match_repr_str(result_match)))

if __name__ == '__main__':
    unittest.main()