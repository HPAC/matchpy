# -*- coding: utf-8 -*-
import doctest
import inspect
import itertools
import unittest

from ddt import data, ddt, unpack
from multiset import Multiset

import patternmatcher.expressions as expressions
from patternmatcher.expressions import (Arity, FrozenExpression, Operation,
                                        Substitution, Symbol, SymbolWildcard,
                                        Variable, Wildcard, freeze, unfreeze)

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
f_ac = Operation.new('f', Arity.variadic, associative=True, commutative=True)

a = Symbol('a')
b = Symbol('b')
c = Symbol('c')

_ = Wildcard.dot()
x_ = Variable.dot('x')
y_ = Variable.dot('y')
z_ = Variable.dot('z')
xs_ = Variable.symbol('x')
__ = Wildcard.plus()
x__ = Variable.plus('x')
y__ = Variable.plus('y')
z__ = Variable.plus('z')
___ = Wildcard.star()
x___ = Variable.star('x')
y___ = Variable.star('y')
z___ = Variable.star('z')


@ddt
class ExpressionTest(unittest.TestCase):
    @unpack
    @data(
        (f_i(a),                        a),
        (f_i(a, b),                     f_i(a, b)),
        (f_i(_),                        _),
        (f_i(___),                      f_i(___)),
        (f_i(__),                       f_i(__)),
        (f_i(x_),                       x_),
        (f_i(x___),                     f_i(x___)),
        (f_i(x__),                      f_i(x__)),
        (f_a(f_a(a)),                   f_a(a)),
        (f_a(f_a(a, b)),                f_a(a, b)),
        (f_a(a, f_a(b)),                f_a(a, b)),
        (f_a(f_a(a), b),                f_a(a, b)),
        (f_a(f(a)),                     f_a(f(a))),
        (f_c(a, b),                     f_c(a, b)),
        (f_c(b, a),                     f_c(a, b)),
    )
    def test_operation_simplify(self, initial, simplified):
        self.assertEqual(initial, simplified)

    @unpack
    @data(
        (Operation.new('f', Arity.unary),                       [],         ValueError),
        (Operation.new('f', Arity.unary),                       [a, b],     ValueError),
        (Operation.new('f', Arity.variadic),                    [],         None),
        (Operation.new('f', Arity.variadic),                    [a],        None),
        (Operation.new('f', Arity.variadic),                    [a, b],     None),
        (Operation.new('f', Arity.binary, associative=True),    [a, a, b],  ValueError),
        (Operation.new('f', Arity.binary),                      [x_, x___], ValueError),
        (Operation.new('f', Arity.binary),                      [x_, x_],   None),
    )
    def test_operation_errors(self, operation, operands, error):
        if error is not None:
            with self.assertRaises(error):
                _ = operation(*operands)
        else:
            _ = operation(*operands)

    @unpack
    @data(
        (a,         True),
        (x_,        False),
        (_,         False),
        (f(a),      True),
        (f(a, b),   True),
        (f(x_),     False),
    )
    def test_is_constant(self, expr, expected):
        self.assertEqual(expr.is_constant, expected)

    @unpack
    @data(
        (a,             True),
        (x_,            True),
        (_,             True),
        (x___,          False),
        (___,           False),
        (x__,           False),
        (__,            False),
        (f(a),          True),
        (f(a, b),       True),
        (f(x_),         True),
        (f(x__),        False),
        (f_a(a),        False),
        (f_a(a, b),     False),
        (f_a(x_),       False),
        (f_a(x__),      False),
        (f_c(a),        False),
        (f_c(a, b),     False),
        (f_c(x_),       False),
        (f_c(x__),      False),
        (f_ac(a),       False),
        (f_ac(a, b),    False),
        (f_ac(x_),      False),
        (f_ac(x__),     False),
    )
    def test_is_syntactic(self, expr, expected):
        self.assertEqual(expr.is_syntactic, expected)

    @unpack
    @data(
        (a,                 True),
        (x_,                True),
        (_,                 True),
        (f(a),              True),
        (f(a, b),           True),
        (f(x_),             True),
        (f(x_, x_),         False),
        (f(x_, y_),         True),
        (f(x_, _),          True),
        (f(_, _),           True),
        (f(x_, f(x_)),      False),
        (f(x_, a, f(x_)),   False),
    )
    def test_is_linear(self, expr, expected):
        self.assertEqual(expr.is_linear, expected)

    @unpack
    @data(
        (a,                 ['a']),
        (x_,                []),
        (_,                 []),
        (f(a),              ['a', 'f']),
        (f(a, b),           ['a', 'b', 'f']),
        (f(x_),             ['f']),
        (f(a, a),           ['a', 'a', 'f']),
        (f(f(a), f(b, c)),  ['a', 'b', 'c', 'f', 'f', 'f']),
    )
    def test_symbols(self, expr, expected):
        self.assertEqual(expr.symbols, Multiset(expected))

    @unpack
    @data(
        (a,                     []),
        (x_,                    ['x']),
        (_,                     []),
        (f(a),                  []),
        (f(x_),                 ['x']),
        (f(x_, x_),             ['x', 'x']),
        (f(x_, a),              ['x']),
        (f(x_, a, y_),          ['x', 'y']),
        (f(f(x_), f(b, x_)),    ['x', 'x']),
    )
    def test_variables(self, expr, expected):
        self.assertEqual(expr.variables, Multiset(expected))

    @unpack
    @data(
        (f(a, x_),      None,                       [(f(a, x_),     tuple()),
                                                     (a,            (0, )),
                                                     (x_,           (1, )),
                                                     (_,            (1, 0))]),
        (f(a, f(x_)),   lambda e: e.head is None,   [(x_,           (1, 0)),
                                                     (_,            (1, 0, 0))]),
        (f(a, f(x_)),   lambda e: e.head == f,      [(f(a, f(x_)),  tuple()),
                                                     (f(x_),        (1, ))])
    )
    def test_preorder_iter(self, expr, predicate, expected_result):
        result = list(expr.preorder_iter(predicate))
        self.assertListEqual(result, expected_result)

    GETITEM_EXPR = f(a, f(x_, b), _)

    @unpack
    @data(
        (tuple(),       GETITEM_EXPR),
        ((0, ),         a),
        ((0, 0),        IndexError),
        ((1, ),         f(x_, b)),
        ((1, 0),        x_),
        ((1, 0, 0),     _),
        ((1, 0, 1),     IndexError),
        ((1, 1),        b),
        ((1, 1, 0),     IndexError),
        ((1, 2),        IndexError),
        ((2, ),         _),
        ((3, ),         IndexError),
    )
    def test_getitem(self, pos, expected_result):
        if inspect.isclass(expected_result) and issubclass(expected_result, Exception):
            with self.assertRaises(expected_result):
                _ = self.GETITEM_EXPR[pos]
        else:
            result = self.GETITEM_EXPR[pos]
            self.assertEqual(result, expected_result)

    @unpack
    @data(
        (a,         b,          True),
        (a,         a,          False),
        (a,         x_,         True),
        (x_,        y_,         True),
        (x_,        x_,         False),
        (x__,       x_,         False),
        (x_,        x__,        False),
        (f(a),      f(b),       True),
        (f(a),      f(a),       False),
        (f(b),      f(a, a),    True),
        (f(a),      f(a, a),    True),
        (f(a, a),   f(a, b),    True),
        (f(a, a),   f(a, a),    False),
        (a,         f(a),       True),
        (x_,        f(a),       True),
        (_,         f(a),       True),
        (x_,        _,          True),
        (a,         _,          True),
    )
    def test_lt(self, expr1, expr2, is_bigger):
        if is_bigger:
            self.assertTrue(expr1 < expr2, '%s < %s did not hold' % (expr1, expr2))
            self.assertFalse(expr2 < expr1, '%s < %s but should not be' % (expr2, expr1))
        else:
            self.assertFalse(expr1 < expr2, '%s < %s but should not be' % (expr1, expr2))

    def test_from_args(self):
        expr = f.from_args(a, b)
        self.assertEqual(expr, f(a, b))

    def test_operation_new_error(self):
        with self.assertRaises(ValueError):
            _ = Operation.new('if', Arity.variadic)

        with self.assertRaises(ValueError):
            _ = Operation.new('+', Arity.variadic)

    def test_variable_error(self):
        with self.assertRaises(ValueError):
            _ = Variable('x', Variable.fixed('y', 2))

        with self.assertRaises(ValueError):
            _ = Variable('x', a)

    def test_wildcard_error(self):
        with self.assertRaises(ValueError):
            _ = Wildcard(-1, False)

        with self.assertRaises(ValueError):
            _ = Wildcard(0, True)

    def test_symbol_wildcard_error(self):
        with self.assertRaises(TypeError):
            _ = SymbolWildcard(object)


@ddt
class FrozenExpressionTest(unittest.TestCase):
    BUILTIN_PROPERTIES = ['is_constant', 'is_syntactic', 'is_linear', 'symbols', 'variables']

    @data(
        a,
        b,
        f(a, b),
        x_,
        ___,
        Variable('x', f(_)),
        xs_
    )
    def test_freeze_eq(self, expr):
        frozen_expr = freeze(expr)
        self.assertEqual(expr, frozen_expr)
        for attr in itertools.chain(vars(expr), self.BUILTIN_PROPERTIES):
            if attr == 'operands':
                self.assertEqual(getattr(frozen_expr, attr), tuple(getattr(expr, attr)), "Operands of frozen instance differs")
            else:
                self.assertEqual(getattr(frozen_expr, attr), getattr(expr, attr), "Attribute %s of frozen instance differs" % attr)

        refrozen = freeze(frozen_expr)
        self.assertIs(refrozen, frozen_expr)

    @data(
        a,
        b,
        f(a, b),
        x_,
        ___,
        Variable('x', f(_)),
        xs_
    )
    def test_unfreeze(self, expr):
        unfrozen = unfreeze(freeze(expr))
        self.assertEqual(unfrozen, expr)

        self.assertIs(expr, unfreeze(expr))

    def test_from_args(self):
        frozen = freeze(f(a))
        expr = type(frozen).from_args(a, b)
        self.assertEqual(expr, f(a, b))
        self.assertIsInstance(expr, FrozenExpression)

    @unpack
    @data(*itertools.product([
        a,
        b,
        f(a, b),
        x_,
        ___,
        Variable('x', f(_)),
        xs_
    ], repeat=2))
    def test_hash(self, expr, other):
        frozen = freeze(expr)
        other = freeze(other)
        if expr != other:
            self.assertNotEqual(hash(frozen), hash(other), 'hash(%s) == hash(%s)' % (frozen, other))
        else:
            self.assertEqual(hash(frozen), hash(other), 'hash(%s) != hash(%s)' % (frozen, other))

    def test_change_error(self):
        frozen = freeze(f(a))

        with self.assertRaises(TypeError):
            frozen.operands[0] = b

        with self.assertRaises(TypeError):
            frozen.operands = [a, b]


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(expressions))
    return tests

if __name__ == '__main__':
    unittest.main()
