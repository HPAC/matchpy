# -*- coding: utf-8 -*-
import unittest
import doctest

from ddt import data, ddt, unpack

from patternmatcher.expressions import Substitution, Symbol, freeze
import patternmatcher.expressions as expressions

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


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(expressions))
    return tests

if __name__ == '__main__':
    unittest.main()