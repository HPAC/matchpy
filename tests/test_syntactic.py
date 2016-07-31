# -*- coding: utf-8 -*-
import unittest
from ddt import ddt, data, unpack

from patternmatcher.expressions import Operation, Symbol, Variable, Arity, Wildcard
from patternmatcher.syntactic import flatterm_iter, OPERATION_END as OP_END

f = Operation.new('f', Arity.variadic)
g = Operation.new('g', Arity.variadic)
h = Operation.new('h', Arity.variadic)
a = Symbol('a')
b = Symbol('b')
c = Symbol('c')
w = Wildcard.dot()
x = Variable.dot('x')
y = Wildcard.plus()
z = Variable.plus('z')

@ddt
class FlattermIterTest(unittest.TestCase):
    """Unit tests for :function:`flatterm_iter`"""

    @unpack
    @data(
        (a,                 [a]),
        (w,                 [w]),
        (x,                 [w]),
        (Variable('v', a),  [a]),
        (f(),               [f, OP_END]),
        (f(a),              [f, a, OP_END]),
        (g(b),              [g, b, OP_END]),
        (f(a, b),           [f, a, b, OP_END]),
        (f(x),              [f, w, OP_END]),
        (f(y),              [f, y, OP_END]),
        (f(g(a)),           [f, g, a, OP_END, OP_END]),
        (f(g(a), b),        [f, g, a, OP_END, b, OP_END]),
        (f(a, g(b)),        [f, a, g, b, OP_END, OP_END]),
        (f(a, g(b), c),     [f, a, g, b, OP_END, c, OP_END]),
        (f(g(b), g(c)),     [f, g, b, OP_END, g, c, OP_END, OP_END]),
        (f(f(g(b)), g(c)),  [f, f, g, b, OP_END, OP_END, g, c, OP_END, OP_END])
    )
    def test_iter(self, expr, result):
        term = list(flatterm_iter(expr))
        self.assertEqual(term, result)

    def test_error(self):
        with self.assertRaises(TypeError):
            flatterm_iter(None).__next__()

if __name__ == '__main__':
    unittest.main()