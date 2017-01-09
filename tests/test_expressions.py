# -*- coding: utf-8 -*-
import inspect
import itertools
from unittest.mock import Mock

import pytest
from multiset import Multiset

from matchpy.expressions import (Arity, FrozenExpression, Operation,
                                        Substitution, Symbol, SymbolWildcard,
                                        Variable, Wildcard, freeze, unfreeze)

from .utils import MockConstraint

a = freeze(Symbol('a'))
b = freeze(Symbol('b'))


f = Operation.new('f', Arity.variadic)
f_i = Operation.new('f_i', Arity.variadic, one_identity=True)
f_a = Operation.new('f_a', Arity.variadic, associative=True)
f_c = Operation.new('f_c', Arity.variadic, commutative=True)
f_ac = Operation.new('f', Arity.variadic, associative=True, commutative=True)

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


class TestSubstitution:
    @pytest.mark.parametrize(
        '   substitution,                   variable,   value,                 expected_result',
        [
            ({},                            'x',        a,                     {'x': a}),
            ({'x': a},                      'x',        a,                     {'x': a}),
            ({'x': a},                      'x',        b,                     ValueError),
            ({'x': a},                      'x',        (a, b),                ValueError),
            ({'x': (a, b)},                 'x',        (a, b),                {'x': (a, b)}),
            ({'x': (a, b)},                 'x',        (a, a),                ValueError),
            ({'x': (a, b)},                 'x',        Multiset([a, b]),      {'x': (a, b)}),
            ({'x': (a, b)},                 'x',        Multiset([a]),         ValueError),
            ({'x': Multiset([a, b])},       'x',        Multiset([a, b]),      {'x': Multiset([a, b])}),
            ({'x': Multiset([a, b])},       'x',        Multiset([]),          ValueError),
            ({'x': Multiset([a, b])},       'x',        (a, b),                {'x': (a, b)}),
            ({'x': Multiset([a, b])},       'x',        (a, a),                ValueError),
            ({'x': Multiset([a])},          'x',        (a,),                  {'x': (a,)}),
            ({'x': Multiset([a])},          'x',        (b,),                  ValueError),
            ({'x': Multiset([a])},          'x',        a,                     {'x': a}),
            ({'x': Multiset([a])},          'x',        b,                     ValueError),
        ]
    )
    def test_union_with_var(self, substitution, variable, value, expected_result):
        substitution = Substitution(substitution)
        if expected_result is ValueError:
            with pytest.raises(ValueError):
                _ = substitution.union_with_variable(variable, value)
        else:
            result = substitution.union_with_variable(variable, value)
            assert result == expected_result

    @pytest.mark.parametrize(
        '   substitution1,                  substitution2,                  expected_result',
        [
            ({},                            {},                             {}),
            ({'x': a},                      {},                             {'x': a}),
            ({'x': a},                      {'y': b},                       {'x': a, 'y': b}),
            ({'x': a},                      {'x': b},                       ValueError),
            ({'x': a},                      {'x': a},                       {'x': a}),
        ]
    )
    def test_union(self, substitution1, substitution2, expected_result):
        substitution1 = Substitution(substitution1)
        substitution2 = Substitution(substitution2)
        if expected_result is ValueError:
            with pytest.raises(ValueError):
                _ = substitution1.union(substitution2)
            with pytest.raises(ValueError):
                _ = substitution2.union(substitution1)
        else:
            result = substitution1.union(substitution2)
            assert result == expected_result
            assert result is not substitution1
            assert result is not substitution2
            result = substitution2.union(substitution1)
            assert result == expected_result
            assert result is not substitution1
            assert result is not substitution2

    @pytest.mark.parametrize(
        '   substitution,                   subject,    pattern,                expected_result',
        [
            ({},                            a,          a,                      {}),
            ({},                            a,          x_,                     {'x': a}),
            ({'x': a},                      a,          x_,                     {'x': a}),
            ({'x': b},                      a,          x_,                     False),
            ({},                            f(a),       f(a),                   {}),
            ({},                            f(a),       f(x_),                  {'x': a}),
            ({'x': a},                      f(a),       f(x_),                  {'x': a}),
            ({'x': b},                      f(a),       f(x_),                  False),
            ({},                            f(a, a),    f(x_, x_),              {'x': a}),
            ({},                            f(a, b),    f(x_, x_),              False),
            ({},                            f(a, b),    f(x_, y_),              {'x': a, 'y': b}),
        ]
    )
    def test_extract_substitution(self, substitution, subject, pattern, expected_result):
        substitution = Substitution(substitution)
        if expected_result is False:
            assert substitution.extract_substitution(subject, pattern) is False
        else:
            assert substitution.extract_substitution(subject, pattern) is True
            assert substitution == expected_result

    @pytest.mark.parametrize(
        '   substitution,                  renaming,                  expected_result',
        [
            ({},                            {},                       {}),
            ({'x': a},                      {},                       {'x': a}),
            ({'x': a},                      {'x': 'y'},               {'y': a}),
            ({'x': a},                      {'y': 'x'},               {'x': a}),
        ]
    )
    def test_rename(self, substitution, renaming, expected_result):
        assert Substitution(substitution).rename(renaming) == expected_result


a = Symbol('a')
b = Symbol('b')

constraint1 = MockConstraint(True)
constraint2 = MockConstraint(True)


class TestExpression:
    @pytest.mark.parametrize(
        '   expression,                     simplified',
        [
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
        ]
    )
    def test_operation_simplify(self, expression, simplified):
        assert expression == simplified

    @pytest.mark.parametrize(
        '   operation,                                              operands,   expected_error',
        [
            (Operation.new('f', Arity.unary),                       [],         ValueError),
            (Operation.new('f', Arity.unary),                       [a, b],     ValueError),
            (Operation.new('f', Arity.variadic),                    [],         None),
            (Operation.new('f', Arity.variadic),                    [a],        None),
            (Operation.new('f', Arity.variadic),                    [a, b],     None),
            (Operation.new('f', Arity.binary, associative=True),    [a, a, b],  ValueError),
            (Operation.new('f', Arity.binary),                      [x_, x___], ValueError),
            (Operation.new('f', Arity.binary),                      [x_, x_],   None),
        ]
    )
    def test_operation_errors(self, operation, operands, expected_error):
        if expected_error is not None:
            with pytest.raises(expected_error):
                operation(*operands)
        else:
            _ = operation(*operands)

    @pytest.mark.parametrize(
        '   expression,     is_constant',
        [
            (a,             True),
            (x_,            False),
            (_,             False),
            (f(a),          True),
            (f(a, b),       True),
            (f(x_),         False),
        ]
    )
    def test_is_constant(self, expression, is_constant):
        assert expression.is_constant == is_constant

    @pytest.mark.parametrize(
        '   expression,     is_syntactic',
        [
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
        ]
    )
    def test_is_syntactic(self, expression, is_syntactic):
        assert expression.is_syntactic == is_syntactic

    @pytest.mark.parametrize(
        '   expression,         is_linear',
        [
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
        ]
    )
    def test_is_linear(self, expression, is_linear):
        assert expression.is_linear == is_linear

    @pytest.mark.parametrize(
        '   expression,         symbols',
        [
            (a,                 ['a']),
            (x_,                []),
            (_,                 []),
            (f(a),              ['a', 'f']),
            (f(a, b),           ['a', 'b', 'f']),
            (f(x_),             ['f']),
            (f(a, a),           ['a', 'a', 'f']),
            (f(f(a), f(b, c)),  ['a', 'b', 'c', 'f', 'f', 'f']),
        ]
    )
    def test_symbols(self, expression, symbols):
        assert expression.symbols == Multiset(symbols)

    @pytest.mark.parametrize(
        '   expression,             variables',
        [
            (a,                     []),
            (x_,                    ['x']),
            (_,                     []),
            (f(a),                  []),
            (f(x_),                 ['x']),
            (f(x_, x_),             ['x', 'x']),
            (f(x_, a),              ['x']),
            (f(x_, a, y_),          ['x', 'y']),
            (f(f(x_), f(b, x_)),    ['x', 'x']),
        ]
    )
    def test_variables(self, expression, variables):
        assert expression.variables == Multiset(variables)

    @pytest.mark.parametrize(
        '   expression,     predicate,                  preorder_list',
        [                                               # expression        position
            (f(a, x_),      None,                       [(f(a, x_),         ()),
                                                         (a,                (0, )),
                                                         (x_,               (1, )),
                                                         (_,                (1, 0))]),
            (f(a, f(x_)),   lambda e: e.head is None,   [(x_,               (1, 0)),
                                                         (_,                (1, 0, 0))]),
            (f(a, f(x_)),   lambda e: e.head == f,      [(f(a, f(x_)),      ()),
                                                         (f(x_),            (1, ))])
        ])
    def test_preorder_iter(self, expression, predicate, preorder_list):
        result = list(expression.preorder_iter(predicate))
        assert result == preorder_list

    GETITEM_TEST_EXPRESSION = f(a, f(x_, b), _)

    @pytest.mark.parametrize(
        '   position,       expected_result',
        [
            ((),            GETITEM_TEST_EXPRESSION),
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
        ])
    def test_getitem(self, position, expected_result):
        if inspect.isclass(expected_result) and issubclass(expected_result, Exception):
            with pytest.raises(expected_result):
                _ = self.GETITEM_TEST_EXPRESSION[position]
        else:
            result = self.GETITEM_TEST_EXPRESSION[position]
            assert result == expected_result

    @pytest.mark.parametrize(
        '   expression1,    expression2,    first_is_bigger_than_second',
        [
            (a,             b,              True),
            (a,             a,              False),
            (a,             x_,             True),
            (x_,            y_,             True),
            (x_,            x_,             False),
            (x__,           x_,             False),
            (x_,            x__,            False),
            (f(a),          f(b),           True),
            (f(a),          f(a),           False),
            (f(b),          f(a, a),        True),
            (f(a),          f(a, a),        True),
            (f(a, a),       f(a, b),        True),
            (f(a, a),       f(a, a),        False),
            (a,             f(a),           True),
            (x_,            f(a),           True),
            (_,             f(a),           True),
            (x_,            _,              True),
            (a,             _,              True),
        ]
    )
    def test_lt(self, expression1, expression2, first_is_bigger_than_second):
        if first_is_bigger_than_second:
            assert expression1 < expression2, "{!s} < {!s} did not hold".format(expression1, expression2)
            assert not (expression2 < expression1), "{!s} < {!s} but should not be".format(expression2, expression1)
        else:
            assert not (expression1 < expression2), "{!s} < {!s} but should not be".format(expression1, expression2)

    def test_from_args(self):
        expression = f.from_args(a, b)
        assert expression == f(a, b)

    def test_operation_new_error(self):
        with pytest.raises(ValueError):
            _ = Operation.new('if', Arity.variadic)

        with pytest.raises(ValueError):
            _ = Operation.new('+', Arity.variadic)

    def test_variable_error(self):
        with pytest.raises(ValueError):
            _ = Variable('x', Variable.fixed('y', 2))

        with pytest.raises(ValueError):
            _ = Variable('x', a)

    def test_wildcard_error(self):
        with pytest.raises(ValueError):
            _ = Wildcard(-1, False)

        with pytest.raises(ValueError):
            _ = Wildcard(0, True)

    def test_symbol_wildcard_error(self):
        with pytest.raises(TypeError):
            _ = SymbolWildcard(object)

    @pytest.mark.parametrize(
        '   expression,                                                     expected_result',
        [
            (a,                                                             a),
            (x_,                                                            x_),
            (Variable.dot('x', constraint1),                                x_),
            (Variable.dot('x', constraint1),                                x_),
            (SymbolWildcard(constraint=constraint1),                        SymbolWildcard()),
            (f(a, constraint=constraint1),                                  f(a)),
            (f(Variable.dot('x', constraint1)),                             f(x_)),
            (f(Variable.dot('x', constraint1), constraint=constraint2),     f(x_)),
        ]
    )
    def test_without_constraints(self, expression, expected_result):
        new_expr = expression.without_constraints
        assert new_expr == expected_result
        assert new_expr is not expression

        frozen_expression = freeze(expression)
        new_expr = frozen_expression.without_constraints
        assert new_expr == expected_result
        assert new_expr is not frozen_expression

    @pytest.mark.parametrize(
        '   expression,                         renaming,       expected_result',
        [
            (a,                                 {},             a),
            (a,                                 {'x': 'y'},     a),
            (x_,                                {},             x_),
            (x_,                                {'x': 'y'},     y_),
            (Variable.dot('x', constraint1),    {'x': 'y'},     Variable.dot('y', constraint1)),
            (SymbolWildcard(),                  {},             SymbolWildcard()),
            (SymbolWildcard(),                  {'x': 'y'},     SymbolWildcard()),
            (f(x_),                             {},             f(x_)),
            (f(x_),                             {'x': 'y'},     f(y_)),
            (f(x_, constraint=constraint1),     {'x': 'y'},     f(y_, constraint=constraint1)),
        ]
    )
    def test_with_renamed_vars(self, expression, renaming, expected_result):
        new_expr = expression.with_renamed_vars(renaming)
        assert new_expr == expected_result
        assert new_expr is not expression

        frozen_expression = freeze(expression)
        new_expr = frozen_expression.with_renamed_vars(renaming)
        assert new_expr == expected_result
        assert new_expr is not frozen_expression


class CustomSymbolWithDict(Symbol):
    def __init__(self, name):
        super().__init__(name)
        self.custom = 42


class CustomSymbolWithoutDict(Symbol):
    __slots__ = ('custom', )

    def __init__(self, name):
        super().__init__(name)
        self.custom = 42

class TestFrozenExpression:
    BUILTIN_PROPERTIES = ['is_constant', 'is_syntactic', 'is_linear', 'symbols', 'variables']

    SIMPLE_EXPRESSIONS = [
        a,
        b,
        f(a, b),
        x_,
        ___,
        Variable('x', f(_)),
        xs_,
        CustomSymbolWithDict('custom1'),
        CustomSymbolWithoutDict('custom2')
    ]

    @pytest.mark.parametrize('expression', SIMPLE_EXPRESSIONS)
    def test_freeze_equivalent(self, expression):
        frozen_expr = freeze(expression)
        assert expression == frozen_expr
        slots = set().union(*(getattr(cls, '__slots__', []) for cls in type(expression).__mro__))
        if hasattr(expression, '__dict__'):
            slots.update(expression.__dict__.keys())
        for attr in itertools.chain(slots, self.BUILTIN_PROPERTIES):
            if attr == 'operands':
                assert getattr(frozen_expr, attr) == tuple(getattr(expression, attr)), "Operands of frozen instance differs"
            else:
                assert getattr(frozen_expr, attr) == getattr(expression, attr), "Attribute {!s} of frozen instance differs".format(attr)

    @pytest.mark.parametrize('expression', SIMPLE_EXPRESSIONS)
    def test_refreeze(self, expression):
        frozen_expr = freeze(expression)
        refrozen = freeze(frozen_expr)
        assert refrozen is frozen_expr

    @pytest.mark.parametrize('expression', SIMPLE_EXPRESSIONS)
    def test_unfreeze(self, expression):
        unfrozen = unfreeze(freeze(expression))
        assert unfrozen == expression
        assert expression is unfreeze(expression)

    def test_from_args(self):
        frozen = freeze(f(a))
        expression = type(frozen).from_args(a, b)
        assert expression == f(a, b)
        assert isinstance(expression, FrozenExpression)

    @pytest.mark.parametrize('expression', SIMPLE_EXPRESSIONS)
    @pytest.mark.parametrize('other', SIMPLE_EXPRESSIONS)
    def test_hash(self, expression, other):
        frozen = freeze(expression)
        other = freeze(other)
        if expression != other:
            assert hash(frozen) != hash(other), "hash({!s}) == hash({!s})".format(frozen, other)
        else:
            assert hash(frozen) == hash(other), "hash({!s}) != hash({!s})".format(frozen, other)

    def test_immutability(self):
        frozen = freeze(f(a))

        with pytest.raises(TypeError):
            frozen.operands[0] = b

        with pytest.raises(TypeError):
            frozen.operands = [a, b]

if __name__ == '__main__':
    import matchpy.expressions as tested_module
    pytest.main(['--doctest-modules', __file__, tested_module.__file__])
