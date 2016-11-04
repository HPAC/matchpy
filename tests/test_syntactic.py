# -*- coding: utf-8 -*-
import itertools
import unittest

from ddt import data, ddt, unpack

from patternmatcher.expressions import (Arity, Operation, Symbol, Variable,
                                        Wildcard, freeze)
from patternmatcher.syntactic import OPERATION_END as OP_END
from patternmatcher.syntactic import DiscriminationNet, FlatTerm, is_operation

f = Operation.new('f', Arity.variadic)
g = Operation.new('g', Arity.variadic)
h = Operation.new('h', Arity.variadic)
a = Symbol('a')
b = Symbol('b')
c = Symbol('c')
d = Symbol('d')
_ = Wildcard.dot()
x_ = Variable.dot('x')
__ = Wildcard.plus()
y__ = Variable.plus('y')
___ = Wildcard.star()
z___ = Variable.star('z')
_s = Wildcard.symbol()

@ddt
class FlattermTest(unittest.TestCase):
    """Unit tests for :class:`FlatTerm`"""

    @unpack
    @data(
        (a,                     [a]),
        (_,                     [_]),
        (x_,                    [_]),
        (_s,                    [Symbol]),
        (Variable('v', f(_)),   [f, _, OP_END]),
        (f(),                   [f, OP_END]),
        (f(a),                  [f, a, OP_END]),
        (g(b),                  [g, b, OP_END]),
        (f(a, b),               [f, a, b, OP_END]),
        (f(x_),                 [f, _, OP_END]),
        (f(__),                 [f, __, OP_END]),
        (f(g(a)),               [f, g, a, OP_END, OP_END]),
        (f(g(a), b),            [f, g, a, OP_END, b, OP_END]),
        (f(a, g(b)),            [f, a, g, b, OP_END, OP_END]),
        (f(a, g(b), c),         [f, a, g, b, OP_END, c, OP_END]),
        (f(g(b), g(c)),         [f, g, b, OP_END, g, c, OP_END, OP_END]),
        (f(f(g(b)), g(c)),      [f, f, g, b, OP_END, OP_END, g, c, OP_END, OP_END]),
        (f(_, _),               [f, Wildcard.dot(2), OP_END]),
        (f(_, __),              [f, Wildcard(2, False), OP_END]),
        (f(_, __, __),          [f, Wildcard(3, False), OP_END]),
        (f(_, ___),             [f, __, OP_END]),
        (f(___, _),             [f, __, OP_END]),
        (f(___, __),            [f, __, OP_END]),
        (f(_, a, __),           [f, _, a, __, OP_END]),
    )
    def test_iter(self, expr, result):
        term = list(FlatTerm(expr))
        self.assertEqual(term, result)

    def test_error(self):
        with self.assertRaises(TypeError):
            FlatTerm(None)

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

@ddt
class DiscriminationNetTest(unittest.TestCase):
    """Unit tests for :method:`DiscriminationNet._generate_net`"""

    @unpack
    @data(
        (a,                         a,                                      True),
        (_,                         a,                                      True),
        (_,                         g(a),                                   True),
        (_,                         g(h(a)),                                True),
        (_,                         g(a, b),                                True),
        (f(_),                      f(a),                                   True),
        (f(_),                      f(g(a)),                                True),
        (f(_),                      f(g(h(a))),                             True),
        (f(_),                      f(g(a, b)),                             True),
        (f(a, a),                   f(a),                                   False),
        (f(a, a),                   f(a, a),                                True),
        (f(a, a),                   f(a, b),                                False),
        (f(a, a),                   f(b, a),                                False),
        (f(__, a),                  f(a),                                   False),
        (f(__, a),                  f(a, a),                                True),
        (f(__, a),                  f(a, b, a),                             True),
        (f(__, a, a),               f(a, b, a, a),                          True),
        (f(__, a, a),               f(a, a, a),                             True),
        (f(__, a, a),               f(a),                                   False),
        (f(__, a, a),               f(a, a),                                False),
        (f(___, a),                 f(a),                                   True),
        (f(___, a),                 f(a, a),                                True),
        (f(___, a),                 f(a, b, a),                             True),
        (f(___, a, a),              f(a, b, a, a),                          True),
        (f(___, a, a),              f(a, a, a),                             True),
        (f(___, a, a),              f(a),                                   False),
        (f(___, a, a),              f(a, a),                                True),
        (f(___, a, b),              f(a, b, a, a),                          False),
        (f(___, a, b),              f(a, b, a, b),                          True),
        (f(___, a, b),              f(a, a, b),                             True),
        (f(___, a, b),              f(a, b, b),                             False),
        (f(___, a, b),              f(a, b),                                True),
        (f(___, a, b),              f(a, a),                                False),
        (f(___, a, b),              f(b, b),                                False),
        (f(___, a, _),              f(a),                                   False),
        (f(___, a, _),              f(a, a),                                True),
        (f(___, a, _),              f(a, b),                                True),
        (f(___, a, _),              f(a, b, a),                             False),
        (f(___, a, __, a),          f(a, b, a),                             True),
        (f(___, a, __, a),          f(a, a, a),                             True),
        (f(___, a, __, a),          f(a, b, a, b),                          False),
        (f(___, a, __, a),          f(b, b, a, a),                          False),
        (f(___, a, __, a),          f(b, a, a, a),                          True),
        (f(___, a, __, a),          f(a, b, b, a),                          True),
        (f(___, g(a)),              f(g(a)),                                True),
        (f(___, g(a)),              f(a, g(a)),                             True),
        (f(___, g(a)),              f(g(a), g(a)),                          True),
        (f(___, g(a)),              f(g(a), g(b)),                          False),
        (f(___, g(a)),              f(g(b), g(a)),                          True),
        (f(___, g(_)),              f(g(a), g(b)),                          True),
        (f(___, g(_)),              f(g(b), g(a)),                          True),
        (f(___, g(_)),              f(g(b), g(h(a))),                       True),
        (f(___, g(_)),              f(g(b), g(h(a, b))),                    True),
        (f(___, g(_)),              f(g(b), g(h(a), a)),                    False),
        (f(___, g(___)),            f(g(a), g(b)),                          True),
        (f(___, g(___)),            f(g(b), g(a, b)),                       True),
        (f(___, g(___, a)),         f(g(a), g(b)),                          False),
        (f(___, g(___, a)),         f(g(b), g(a, b)),                       False),
        (f(___, g(___, a)),         f(g(b), g(a, a)),                       True),
        (f(___, g(___, a)),         f(g(a, b), g(a, a)),                    True),
        (f(___, g(___, a)),         f(g(b, b), g(b, a)),                    True),
        (f(___, g(___, a), b),      f(g(b, a), b, g(b, a)),                 False),
        (f(___, g(___, a), b),      f(g(b, a), b, g(b, a), b),              True),
        (f(___, g(h(a))),           f(g(a)),                                False),
        (f(___, g(h(a))),           f(g(h(b))),                             False),
        (f(___, g(h(a))),           f(g(h(a), b)),                          False),
        (f(___, g(h(a))),           f(g(h(a))),                             True),
        #(f(___, g(h(___, a))),      f(g(a), g(h(b)), g(h(a), b), g(h(a))),  True),
    )
    def test_generate_and_match_correctness(self, pattern, expr, is_match):
        net = DiscriminationNet()
        net._net = DiscriminationNet._generate_net(freeze(pattern))
        result = net.match(freeze(expr))

        if is_match:
            self.assertListEqual([pattern], result)
        else:
            self.assertListEqual([], result)

    def test_match_error(self):
        net = DiscriminationNet()
        pattern = freeze(f(x_))
        net.add(pattern)

        with self.assertRaises(TypeError):
            _ = net.match(pattern)



    pattern_symbols = [a, b] #[a, b, c, _, __, ___]
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

if __name__ == '__main__':
    unittest.main()
