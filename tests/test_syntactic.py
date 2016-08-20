# -*- coding: utf-8 -*-
import unittest
import itertools
from ddt import ddt, data, unpack

from patternmatcher.expressions import Operation, Symbol, Variable, Arity, Wildcard
from patternmatcher.syntactic import Flatterm, OPERATION_END as OP_END, DiscriminationNet, is_operation

f = Operation.new('f', Arity.variadic)
g = Operation.new('g', Arity.variadic)
h = Operation.new('h', Arity.variadic)
a = Symbol('a')
b = Symbol('b')
c = Symbol('c')
d = Symbol('d')
w = Wildcard.dot()
x = Variable.dot('x')
y = Wildcard.plus()
z = Variable.plus('z')
t = Wildcard.star()
s = Variable.star('s')

@ddt
class FlattermTest(unittest.TestCase):
    """Unit tests for :class:`Flatterm`"""

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
        term = list(Flatterm._flatterm_iter(expr))
        self.assertEqual(term, result)

    def test_error(self):
        with self.assertRaises(TypeError):
            Flatterm(None)

def product(iter_factory, repeat):
    iters = [iter_factory() for _ in range(repeat)]
    values = [None] * repeat
    i = 0
    while True:
        try:
            while i < repeat:
                values[i] = iters[i].__next__()
                i += 1
            yield values
            i -= 1
        except StopIteration:
            iters[i] = iter_factory()
            i -= 1
            if i < 0:
                return

class GenerateNetTest(unittest.TestCase):
    """Unit tests for :method:`DiscriminationNet._generate_net`"""

    pattern_symbols = [a, b] #[a, b, c, w, y, t]
    pattern_operations = [f, g]

    def generate_patterns(self, depth, max_args):
        for symbol in GenerateNetTest.pattern_symbols:
            yield symbol
        if depth > 0:
            for operation in GenerateNetTest.pattern_operations:
                yield operation()
            for n in range(1, max_args+1):
                for args in product(lambda: self.generate_patterns(depth-1,max_args), n):
                    for operation in GenerateNetTest.pattern_operations:
                        yield operation(*args)

    def test_correctness(self):
        # TODO
        pass

if __name__ == '__main__':
    unittest.main()