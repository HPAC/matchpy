# -*- coding: utf-8 -*-
import doctest
import unittest

from ddt import data, ddt, unpack

import patternmatcher.expressions as expressions
from patternmatcher.expressions import (Arity, Operation, Substitution, Symbol,
                                        Variable, Wildcard, freeze)

a = freeze(Symbol('a'))
b = freeze(Symbol('b'))

@ddt
class SubstitutionTest(unittest.TestCase):
    @unpack
    @data(
        ({},                            'x', a,                     {'x': a}),
        ({'x': a},                      'x', a,                     {'x': a}),
        ({'x': a},                      'x', b,                     ValueError),
        ({'x': a},                      'x', (a, b),                ValueError),
        ({'x': (a, b)},                 'x', (a, b),                {'x': (a, b)}),
        ({'x': (a, b)},                 'x', (a, a),                ValueError),
        ({'x': (a, b)},                 'x', {a, b},                {'x': (a, b)}),
        ({'x': (a, b)},                 'x', {a},                   ValueError),
        ({'x': {a, b}},                 'x', {a, b},                {'x': {a, b}}),
        ({'x': {a, b}},                 'x', {a},                   ValueError),
        ({'x': {a, b}},                 'x', (a, b),                {'x': (a, b)}),
        ({'x': {a, b}},                 'x', (a, a),                ValueError),
        ({'x': {a}},                    'x', (a,),                  {'x': (a,)}),
        ({'x': {a}},                    'x', (b,),                  ValueError),
        ({'x': {a}},                    'x', a,                     {'x': a}),
        ({'x': {a}},                    'x', b,                     ValueError),
    )
    def test_union_with_var(self, subst, var, value, expected_result):
        subst = Substitution(subst)

        if expected_result is ValueError:
            with self.assertRaises(ValueError):
                _ = subst.union_with_variable(var, value)
        else:
            result = subst.union_with_variable(var, value)
            self.assertEqual(result, expected_result)

f = Operation.new('f', Arity.variadic)
f_i = Operation.new('f_i', Arity.variadic, one_identity=True)
f_a = Operation.new('f_a', Arity.variadic, associative=True)
f_c = Operation.new('f_c', Arity.variadic, commutative=True)

a = Symbol('a')
b = Symbol('b')

@ddt
class ExpressionTest(unittest.TestCase):
    @unpack
    @data(
        (f_i(a),                       a),
        (f_i(a, b),                    f_i(a, b)),
        (f_i(Wildcard.dot()),          Wildcard.dot()),
        (f_i(Wildcard.star()),         f_i(Wildcard.star())),
        (f_i(Wildcard.plus()),         f_i(Wildcard.plus())),
        (f_i(Variable.dot('x')),       Variable.dot('x')),
        (f_i(Variable.star('x')),      f_i(Variable.star('x'))),
        (f_i(Variable.plus('x')),      f_i(Variable.plus('x'))),
        (f_a(f_a(a)),                  f_a(a)),
        (f_a(f_a(a, b)),               f_a(a, b)),
        (f_a(a, f_a(b)),               f_a(a, b)),
        (f_a(f_a(a), b),               f_a(a, b)),
        (f_a(f(a)),                    f_a(f(a))),
        (f_c(a, b),                    f_c(a, b)),
        (f_c(b, a),                    f_c(a, b)),
    )
    def test_operation_simplify(self, initial, simplified):
        self.assertEqual(initial, simplified)

    @unpack
    @data(
        (Operation.new('f', Arity.unary),                       [],                                         ValueError),
        (Operation.new('f', Arity.unary),                       [a, b],                                     ValueError),
        (Operation.new('f', Arity.variadic),                    [],                                         None),
        (Operation.new('f', Arity.variadic),                    [a],                                        None),
        (Operation.new('f', Arity.variadic),                    [a, b],                                     None),
        (Operation.new('f', Arity.binary, associative=True),    [a, a, b],                                  ValueError),
        (Operation.new('f', Arity.binary),                      [Variable.dot('x'), Variable.star('x')],    ValueError),
        (Operation.new('f', Arity.binary),                      [Variable.dot('x'), Variable.dot('x')],     None),
    )
    def test_operation_errors(self, operation, operands, error):
        if error is not None:
            with self.assertRaises(error):
                _ = operation(*operands)
        else:
            _ = operation(*operands)


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(expressions))
    return tests

if __name__ == '__main__':
    unittest.main()
