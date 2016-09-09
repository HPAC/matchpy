# -*- coding: utf-8 -*-
import unittest

from ddt import data, ddt, unpack

from patternmatcher.expressions import Operation, Symbol, Variable, Arity, Wildcard
from patternmatcher.matching import CommutativePatternsParts

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
_ = Wildcard.dot()
x_ = Variable.dot('x')
x2 = Variable.fixed('x', 2)
y_ = Variable.dot('y')
z_ = Variable.dot('z')
__ = Wildcard.plus()
x__ = Variable.plus('x')
y__ = Variable.plus('y')
z__ = Variable.plus('z')
___ = Wildcard.star()
x___ = Variable.star('x')
y___ = Variable.star('y')
z___ = Variable.star('z')

@ddt
class CommutativePatternsPartsTest(unittest.TestCase):
    @unpack
    @data(
        ([],                          [],             [],             [],                   [],                   []),
        ([a],                         [a],            [],             [],                   [],                   []),
        ([a, b],                      [a, b],         [],             [],                   [],                   []),
        ([x_],                        [],             [],             [],                   [('x', 1)],           []),
        ([x_, y_],                    [],             [],             [],                   [('x', 1), ('y', 1)], []),
        ([x2],                        [],             [],             [],                   [('x', 2)],           []),
        ([f(x_)],                     [],             [f(x_)],        [],                   [],                   []),
        ([f(x_), f(y_)],              [],             [f(x_), f(y_)], [],                   [],                   []),
        ([f(a)],                      [f(a)],         [],             [],                   [],                   []),
        ([f(x__)],                    [],             [],             [],                   [],                   [f(x__)]),
        ([f(a), f(b)],                [f(a), f(b)],   [],             [],                   [],                   []),
        ([x__],                       [],             [],             [('x', 1)],           [],                   []),
        ([x___],                      [],             [],             [('x', 0)],           [],                   []),
        ([x__, y___],                 [],             [],             [('x', 1), ('y', 0)], [],                   []),
        ([fc(x_)],                    [],             [],             [],                   [],                   [fc(x_)]),
        ([fc(x_, a)],                 [],             [],             [],                   [],                   [fc(x_, a)]),
        ([fc(x_, a), fc(x_, b)],      [],             [],             [],                   [],                   [fc(x_, a), fc(x_, b)]),
        ([fc(a)],                     [fc(a)],        [],             [],                   [],                   []),
        ([fc(a), fc(b)],              [fc(a), fc(b)], [],             [],                   [],                   []),
        ([a, x_, x__, f(x_), fc(x_)], [a],            [f(x_)],        [('x', 1)],           [('x', 1)],           [fc(x_)]),
    )
    def test_find_cycle(self, expressions, constant, syntactic, seq_vars, fixed_vars, rest):
        parts = CommutativePatternsParts(*expressions)

        self.assertListEqual(constant, sorted(parts.constant.elements()))
        self.assertListEqual(syntactic, sorted(parts.syntactic.elements()))
        self.assertListEqual(seq_vars, sorted(parts.sequence_variables.elements()))
        self.assertListEqual(fixed_vars, sorted(parts.fixed_variables.elements()))
        self.assertListEqual(rest, sorted(parts.rest.elements()))

        self.assertEqual(sum(c for _, c in seq_vars), parts.sequence_variable_min_length)
        self.assertEqual(sum(c for _, c in fixed_vars), parts.fixed_variable_length)


if __name__ == '__main__':
    unittest.main()