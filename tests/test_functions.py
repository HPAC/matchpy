# -*- coding: utf-8 -*-
import unittest
from ddt import ddt, data, unpack

from patternmatcher.expressions import Operation, Symbol, Variable, Arity, Wildcard
from patternmatcher.functions import match


f = Operation.new('f', Arity.variadic)
g = Operation.new('g', Arity.variadic)
h = Operation.new('h', Arity.variadic, commutative=True)
a = Symbol('a')
b = Symbol('b')
c = Symbol('c')
w = Wildcard.dot()
x = Variable.dot('x')
y = Wildcard.plus()
z = Variable.plus('z')
t = Wildcard.star()
s = Variable.star('s')

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

if __name__ == '__main__':
    unittest.main()