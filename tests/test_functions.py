# -*- coding: utf-8 -*-
import unittest
from ddt import ddt, data, unpack

from patternmatcher.expressions import Operation, Symbol, Variable, Arity, Wildcard
from patternmatcher.functions import match
from patternmatcher.utils import match_repr_str


f = Operation.new('f', Arity.variadic)
g = Operation.new('g', Arity.variadic)
h = Operation.new('h', Arity.variadic, commutative=True)
fa = Operation.new('fa', Arity.variadic, associative=True)
a = Symbol('a')
b = Symbol('b')
c = Symbol('c')
w = Wildcard.dot()
x = Variable.dot('x')
x2 = Variable.dot('x2')
y = Wildcard.plus()
z = Variable.plus('z')
z2 = Variable.plus('z2')
t = Wildcard.star()
s = Variable.star('s')
s2 = Variable.star('s2')

@ddt
class MatchTest(unittest.TestCase):
    @unpack
    @data(
        (a,                 a,          True),
        (b,                 a,          False),
        (f(a),              f(a),       True),
        (f(b),              f(a),       False),
        (g(a),              f(a),       False),
        (f(a, b),           f(a),       False),
        (f(a, b),           f(a, b),    True),
        (f(a),              f(a, b),    False),
        (f(b, a),           f(a, b),    False),
        (f(a, b, c),        f(a, b),    False),
        (f(a, g(b)),        f(a, b),    False),
        (f(g(a), g(b)),     f(a, b),    False),
        (f(g(a), b),        f(a, b),    False),
        (f(f(a, b)),        f(a, b),    False),
        (g(a, b),           f(a, b),    False),
        (f(a, g(b)),        f(a, g(b)), True),
        (f(g(a), b),        f(g(a), b), True),
        (f(f(a, b)),        f(f(a, b)), True)
    )
    def test_constant_match(self, expr, pattern, is_match):
        result = list(match([expr], pattern))
        if is_match:
            self.assertEqual(result, [dict()], 'Expression %s and %s did not match but were supposed to' % (expr, pattern))
        else:
            self.assertEqual(result, [], 'Expression %s and %s did match but were not supposed to' % (expr, pattern))

    @unpack
    @data(
        (h(a, b),        h(a, b),       True),
        (h(b, a),        h(a, b),       True),
        (h(b, a, c),     h(a, b, c),    True),
        (h(c, a, b),     h(a, b, c),    True),
        (h(b, a, c),     h(c, b, a),    True),
        (h(b, a, a),     h(a, a, b),    True),
        (h(a, b, a),     h(a, a, b),    True),
        (h(b, b, a),     h(a, a, b),    False),
        (h(c, a, g(b)),  h(a, g(b), c), True),
        (h(c, a, g(b)),  h(g(a), b, c), False),
        (g(c, h(a, b)),  g(c, h(b, a)), True),
        (g(c, h(a, b)),  g(h(a, b), c), False),
    )
    def test_commutative_match(self, expr, pattern, is_match):
        result = list(match([expr], pattern))
        if is_match:
            self.assertEqual(result, [dict()], 'Expression %s and %s did not match but were supposed to' % (expr, pattern))
        else:
            self.assertEqual(result, [], 'Expression %s and %s did match but were not supposed to' % (expr, pattern))

    @unpack
    @data(
        (a,                 x,              {'x': a}),
        (b,                 x,              {'x': b}),
        (f(a),              f(x),           {'x': a}),
        (f(b),              f(x),           {'x': b}),
        (f(a),              x,              {'x': f(a)}),
        (g(a),              f(x),           None),
        (f(a, b),           f(x),           None),
        (f(a, b),           f(x, b),        {'x': a}),
        (f(a, b),           f(x, a),        None),
        (f(a, b),           f(a, x),        {'x': b}),
        (f(a, b),           f(x, x),        None),
        (f(a, a),           f(x, x),        {'x': a}),
        (f(a, b),           f(x, x2),       {'x': a,    'x2': b}),
        (f(a),              f(x, x2),       None),
        (f(a, b, c),        f(x, x2),       None),
        (f(a, g(b)),        f(x, x2),       {'x': a,    'x2': g(b)}),
        (f(a, g(b)),        f(x, g(x2)),    {'x': a,    'x2': b}),
        (f(a, g(b)),        f(x, g(x)),     None),
        (f(a, g(a)),        f(x, g(x)),     {'x': a}),
        (f(g(a), g(b)),     f(x, x),        None),
        (f(g(a), g(b)),     f(x, x2),       {'x': g(a), 'x2': g(b)}),
        (f(g(a), a),        f(x, x),        None),
        (f(g(a), a),        f(g(x), x),     {'x': a}),
        (f(f(a, b)),        f(x, x2),       None),
        (f(f(a, b)),        f(x),           {'x': f(a, b)}),
        (g(a, b),           f(x, x2),       None),
        (f(f(a, b)),        f(f(x, x2)),    {'x': a,    'x2': b})
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
        (fa(a),                 fa(x),         [{'x': a}]),
        (fa(a, b),              fa(x),         [{'x': fa(a, b)}]),
        (fa(a, b),              fa(a, x),      [{'x': b}]),
        (fa(a, b, c),           fa(a, x),      [{'x': fa(b, c)}]),
        (fa(a, b, c),           fa(x, c),      [{'x': fa(a, b)}]),
        (fa(a, b, c),           fa(x),         [{'x': fa(a, b, c)}]),
        (fa(a, b, a, b),        fa(x, x),      [{'x': fa(a, b)}]),
        (fa(a, b, a),           fa(x, b, x),   [{'x': a}]),
        (fa(a, a, b, a, a),     fa(x, b, x),   [{'x': fa(a, a)}]),
        (fa(a, b, c),           fa(x, x2),     [{'x': a,        'x2': fa(b, c)}, \
                                                {'x': fa(a, b), 'x2': c}])
    )
    def test_associative_wildcard_dot_match(self, expr, pattern, expected_matches):
        result = list(match([expr], pattern))
        for expected_match in expected_matches:
            self.assertIn(expected_match, result, 'Expression %s and %s did not yield the match %s but were supposed to' % (expr, pattern, match_repr_str(expected_match)))
        for result_match in result:
            self.assertIn(result_match, expected_matches, 'Expression %s and %s yielded the unexpected match %s' % (expr, pattern, match_repr_str(result_match)))

    @unpack
    @data(
        (a,                         s,              [{'s': [a]}]),
        (f(a),                      f(s),           [{'s': [a]}]),
        (f(),                       f(s),           [{'s': []}]),
        (f(a),                      s,              [{'s': [f(a)]}]),
        (g(a),                      f(s),           []),
        (f(a, b),                   f(s),           [{'s': [a, b]}]),
        (f(a, b),                   f(s, b),        [{'s': [a]}]),
        (f(a, b),                   f(s, a),        []),
        (f(a, b),                   f(a, s),        [{'s': [b]}]),
        (f(a, b),                   f(s, s),        []),
        (f(a, a),                   f(s, s),        [{'s': [a]}]),
        (f(a, b),                   f(s, s2),       [{'s': [],           's2': [a, b]},     \
                                                     {'s': [a],          's2': [b]},        \
                                                     {'s': [a, b],       's2': []}]),
        (f(a),                      f(s, s2),       [{'s': [],           's2': [a]},        \
                                                     {'s': [a],          's2': []}]),
        (f(a, b, c),                f(s, s2),       [{'s': [],           's2': [a, b, c]},  \
                                                     {'s': [a],          's2': [b, c]},     \
                                                     {'s': [a, b],       's2': [c]},        \
                                                     {'s': [a, b, c],    's2': []}]),
        (f(a, g(b)),                f(s, s2),       [{'s': [],           's2': [a, g(b)]},  \
                                                     {'s': [a],          's2': [g(b)]},     \
                                                     {'s': [a, g(b)],    's2': []}]),
        (f(a, g(b)),                f(s, g(s2)),    [{'s': [a],          's2': [b]}]),
        (f(a, g(b)),                f(s, g(s)),     []),
        (f(a, g(a)),                f(s, g(s)),     [{'s': [a]}]),
        (f(g(a), g(b)),             f(s, s),        []),
        (f(g(a), g(b)),             f(s, s2),       [{'s': [g(a), g(b)], 's2': []},         \
                                                     {'s': [g(a)],       's2': [g(b)]},     \
                                                     {'s': [],           's2': [g(a), g(b)]}]),
        (f(g(a), a),                f(s, s),        []),
        (f(g(a), a),                f(g(s), s),     [{'s': [a]}]),
        (f(f(a, b)),                f(s, s2),       [{'s': [f(a, b)],    's2': []},         \
                                                     {'s': [],           's2': [f(a, b)]}]),
        (f(f(a, b)),                f(s),           [{'s': [f(a, b)]}]),
        (g(a, b),                   f(s, s2),       []),
        (f(a, a, a),                f(s, b, s2),    []),
        (f(a, a, a),                f(s, a, s2),    [{'s': [],           's2': [a, a]},     \
                                                     {'s': [a],          's2': [a]},        \
                                                     {'s': [a, a],       's2': []}]),
        (f(a),                      f(s, a, s2),    [{'s': [],           's2': []}]),
        (f(a, a),                   f(s, a, s2),    [{'s': [a],          's2': []},         \
                                                     {'s': [],           's2': [a]}]),
        (f(a, b, a),                f(s, a, s2),    [{'s': [],           's2': [b, a]},     \
                                                     {'s': [a, b],       's2': []}]),
        (f(a, b, a, b),             f(s, s),        [{'s': [a, b]}]),
        (f(a, b, a, a),             f(s, s),        []),
        (f(a, b, a),                f(s, b, s),     [{'s': [a]}]),
        (f(a, b, a, a),             f(s, b, s),     []),
        (f(a, a, b, a),             f(s, b, s),     []),
        (f(a, b, a, b, a, b, a),    f(s, b, s),     [{'s': [a, b, a]}]),
        (f(a, b, a, b),             f(s, b, s2),    [{'s': [a, b, a],    's2': []},         \
                                                     {'s': [a],          's2': [a, b]}]),
    )
    def test_wildcard_star_match(self, expr, pattern, expected_matches):
        result = list(match([expr], pattern))
        for expected_match in expected_matches:
            self.assertIn(expected_match, result, 'Expression %s and %s did not yield the match %s but were supposed to' % (expr, pattern, match_repr_str(expected_match)))
        for result_match in result:
            self.assertIn(result_match, expected_matches, 'Expression %s and %s yielded the unexpected match %s' % (expr, pattern, match_repr_str(result_match)))
    
    @unpack
    @data(
        (a,                         z,              [{'z': [a]}]),
        (f(a),                      f(z),           [{'z': [a]}]),
        (f(),                       f(z),           []),
        (f(a),                      z,              [{'z': [f(a)]}]),
        (g(a),                      f(z),           []),
        (f(a, b),                   f(z),           [{'z': [a, b]}]),
        (f(a, b),                   f(z, b),        [{'z': [a]}]),
        (f(a, b),                   f(z, a),        []),
        (f(a, b),                   f(a, z),        [{'z': [b]}]),
        (f(a, b),                   f(z, z),        []),
        (f(a, a),                   f(z, z),        [{'z': [a]}]),
        (f(a, b),                   f(z, z2),       [{'z': [a],          'z2': [b]}]),
        (f(a),                      f(z, z2),       []),
        (f(a, b, c),                f(z, z2),       [{'z': [a],          'z2': [b, c]},     \
                                                     {'z': [a, b],       'z2': [c]}]),
        (f(a, g(b)),                f(z, z2),       [{'z': [a],          'z2': [g(b)]}]),
        (f(a, g(b)),                f(z, g(z2)),    [{'z': [a],          'z2': [b]}]),
        (f(a, g(b)),                f(z, g(z)),     []),
        (f(a, g(a)),                f(z, g(z)),     [{'z': [a]}]),
        (f(g(a), g(b)),             f(z, z),        []),
        (f(g(a), g(b)),             f(z, z2),       [{'z': [g(a)],       'z2': [g(b)]}]),
        (f(g(a), a),                f(z, z),        []),
        (f(g(a), a),                f(g(z), z),     [{'z': [a]}]),
        (f(f(a, b)),                f(z, z2),       []),
        (f(f(a, b)),                f(z),           [{'z': [f(a, b)]}]),
        (g(a, b),                   f(z, z2),       []),
        (f(a, a, a),                f(z, b, z2),    []),
        (f(a, a, a),                f(z, a, z2),    [{'z': [a],          'z2': [a]}]),
        (f(a),                      f(z, a, z2),    []),
        (f(a, a),                   f(z, a, z2),    []),
        (f(a, b, a),                f(z, a, z2),    []),
        (f(a, b, a, b),             f(z, z),        [{'z': [a, b]}]),
        (f(a, b, a, a),             f(z, z),        []),
        (f(a, b, a),                f(z, b, z),     [{'z': [a]}]),
        (f(a, b, a, a),             f(z, b, z),     []),
        (f(a, a, b, a),             f(z, b, z),     []),
        (f(a, b, a, b, a, b, a),    f(z, b, z),     [{'z': [a, b, a]}]),
        (f(a, b, a, b),             f(z, b, z2),    [{'z': [a],          'z2': [a, b]}]),
    )
    def test_wildcard_plus_match(self, expr, pattern, expected_matches):
        result = list(match([expr], pattern))
        for expected_match in expected_matches:
            self.assertIn(expected_match, result, 'Expression %s and %s did not yield the match %s but were supposed to' % (expr, pattern, match_repr_str(expected_match)))
        for result_match in result:
            self.assertIn(result_match, expected_matches, 'Expression %s and %s yielded the unexpected match %s' % (expr, pattern, match_repr_str(result_match)))

if __name__ == '__main__':
    unittest.main()