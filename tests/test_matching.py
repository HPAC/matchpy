# -*- coding: utf-8 -*-
import pytest

from matchpy.expressions import (Arity, CustomConstraint,
                                        MultiConstraint, Operation, Symbol,
                                        Variable, Wildcard, freeze)
from matchpy.matching.many_to_one import ManyToOneMatcher
from matchpy.matching.common import CommutativePatternsParts

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

constr1 = CustomConstraint(lambda x, y: x == y)
constr2 = CustomConstraint(lambda x, y: x != y)


class TestCommutativePatternsParts:
    @pytest.mark.parametrize(
        '   expressions,                    constant,       syntactic,      sequence_variables,     fixed_variables,        rest',
        [
            ([],                            [],             [],             [],                     [],                     []),
            ([a],                           [a],            [],             [],                     [],                     []),
            ([a, b],                        [a, b],         [],             [],                     [],                     []),
            ([x_],                          [],             [],             [],                     [('x', 1)],             []),
            ([x_, y_],                      [],             [],             [],                     [('x', 1), ('y', 1)],   []),
            ([x2],                          [],             [],             [],                     [('x', 2)],             []),
            ([f(x_)],                       [],             [f(x_)],        [],                     [],                     []),
            ([f(x_), f(y_)],                [],             [f(x_), f(y_)], [],                     [],                     []),
            ([f(a)],                        [f(a)],         [],             [],                     [],                     []),
            ([f(x__)],                      [],             [],             [],                     [],                     [f(x__)]),
            ([f(a), f(b)],                  [f(a), f(b)],   [],             [],                     [],                     []),
            ([x__],                         [],             [],             [('x', 1)],             [],                     []),
            ([x___],                        [],             [],             [('x', 0)],             [],                     []),
            ([x__, y___],                   [],             [],             [('x', 1), ('y', 0)],   [],                     []),
            ([fc(x_)],                      [],             [],             [],                     [],                     [fc(x_)]),
            ([fc(x_, a)],                   [],             [],             [],                     [],                     [fc(x_, a)]),
            ([fc(x_, a), fc(x_, b)],        [],             [],             [],                     [],                     [fc(x_, a), fc(x_, b)]),
            ([fc(a)],                       [fc(a)],        [],             [],                     [],                     []),
            ([fc(a), fc(b)],                [fc(a), fc(b)], [],             [],                     [],                     []),
            ([a, x_, x__, f(x_), fc(x_)],   [a],            [f(x_)],        [('x', 1)],             [('x', 1)],             [fc(x_)]),
            ([__],                          [],             [],             [],                     [],                     []),
            ([_],                           [],             [],             [],                     [],                     []),
            ([_, _],                        [],             [],             [],                     [],                     []),
            ([___],                         [],             [],             [],                     [],                     []),
            ([___, __, _],                  [],             [],             [],                     [],                     []),
            ([__, x_],                      [],             [],             [],                     [('x', 1)],             []),
            ([__, x__],                     [],             [],             [('x', 1)],             [],             []),
        ]
    )
    def test_parts(self, expressions, constant, syntactic, sequence_variables, fixed_variables, rest):
        parts = CommutativePatternsParts(None, *map(freeze, expressions))

        assert constant == sorted(parts.constant)
        assert syntactic == sorted(parts.syntactic)

        assert len(sequence_variables) == len(parts.sequence_variables)
        for name, min_count in sequence_variables:
            assert name in parts.sequence_variables
            assert name in parts.sequence_variable_infos
            assert min_count == parts.sequence_variable_infos[name].min_count

        assert len(fixed_variables) == len(parts.fixed_variables)
        for name, min_count in fixed_variables:
            assert name in parts.fixed_variables
            assert name in parts.fixed_variable_infos
            assert min_count == parts.fixed_variable_infos[name].min_count

        assert rest == sorted(parts.rest)

        assert sum(c for _, c in sequence_variables) == parts.sequence_variable_min_length
        assert sum(c for _, c in fixed_variables) == parts.fixed_variable_length

        if any(isinstance(o, Wildcard) for o in expressions):
            fixed = all(wc.fixed_size for wc in expressions if isinstance(wc, Wildcard))
            length = sum(wc.min_count for wc in expressions if isinstance(wc, Wildcard))

            assert parts.wildcard_fixed is fixed
            assert parts.wildcard_min_length == length
        else:
            assert parts.wildcard_fixed is None


    @pytest.mark.parametrize(
        '   constraints,                        result_constraint',
        [
            ([None],                            None),
            ([constr1],                         constr1),
            ([constr1,  constr1],               constr1),
            ([None,     constr1],               constr1),
            ([constr1,  None],                  constr1),
            ([None,     None,       constr1],   constr1),
            ([None,     constr1,    None],      constr1),
            ([constr1,  None,       None],      constr1),
            ([constr1,  constr2],               MultiConstraint(constr1, constr2)),
            ([None,     constr1,    constr2],   MultiConstraint(constr1, constr2)),
            ([constr1,  None,       constr2],   MultiConstraint(constr1, constr2)),
            ([constr1,  constr2,    None],      MultiConstraint(constr1, constr2))
        ]
    )
    def test_fixed_var_constraints(self, constraints, result_constraint):
        parts = CommutativePatternsParts(None, *[Variable('x', Wildcard.dot(), c) for c in constraints])

        assert 1 == len(parts.fixed_variables.keys())
        assert len(constraints) == len(parts.fixed_variables)
        assert 'x' in parts.fixed_variables
        assert 'x' in parts.fixed_variable_infos

        info = parts.fixed_variable_infos['x']
        assert 1 == info.min_count
        assert result_constraint == info.constraint

    @pytest.mark.parametrize(
        '   constraints,                        result_constraint',
        [
            ([None],                            None),
            ([constr1],                         constr1),
            ([constr1,  constr1],               constr1),
            ([None,     constr1],               constr1),
            ([constr1,  None],                  constr1),
            ([None,     None,       constr1],   constr1),
            ([None,     constr1,    None],      constr1),
            ([constr1,  None,       None],      constr1),
            ([constr1,  constr2],               MultiConstraint(constr1, constr2)),
            ([None,     constr1,    constr2],   MultiConstraint(constr1, constr2)),
            ([constr1,  None,       constr2],   MultiConstraint(constr1, constr2)),
            ([constr1,  constr2,    None],      MultiConstraint(constr1, constr2))
        ]
    )
    def test_sequence_var_constraints(self, constraints, result_constraint):
        parts = CommutativePatternsParts(None, *[Variable('x', Wildcard.plus(), c) for c in constraints])

        assert 1 == len(parts.sequence_variables.keys())
        assert len(constraints) == len(parts.sequence_variables)
        assert 'x' in parts.sequence_variables
        assert 'x' in parts.sequence_variable_infos

        info = parts.sequence_variable_infos['x']
        assert 1 == info.min_count
        assert result_constraint == info.constraint


class TestAutomaton:
    def test_different_constraints(self):
        c1 = CustomConstraint(lambda x: len(str(x)) > 1)
        c2 = CustomConstraint(lambda x: len(str(x)) == 1)
        pattern1 = f(Variable.dot('x', c1))
        pattern2 = f(Variable.dot('x', c2))
        pattern3 = f(Variable.dot('x', c1), b)
        pattern4 = f(Variable.dot('x', c2), b)
        matcher = ManyToOneMatcher(pattern1, pattern2, pattern3, pattern4)

        subject = f(a)
        results = list(matcher.match(subject))
        assert len(results) == 1
        assert results[0][0] == pattern2
        assert results[0][1] == {'x': a}

        subject = f(Symbol('longer'), b)
        results = sorted(matcher.match(subject))
        assert len(results) == 1
        assert results[0][0] == pattern3
        assert results[0][1] == {'x': Symbol('longer')}

    def test_different_constraints_on_operation(self):
        c1 = CustomConstraint(lambda x: len(str(x)) > 1)
        c2 = CustomConstraint(lambda x: len(str(x)) == 1)
        pattern1 = f(x_, constraint=c1)
        pattern2 = f(x_, constraint=c2)
        pattern3 = f(x_, b, constraint=c1)
        pattern4 = f(x_, b, constraint=c2)
        matcher = ManyToOneMatcher(pattern1, pattern2, pattern3, pattern4)

        subject = f(a)
        results = list(matcher.match(subject))
        assert len(results) == 1
        assert results[0][0] == pattern2
        assert results[0][1] == {'x': a}

        subject = f(Symbol('longer'), b)
        results = sorted(matcher.match(subject))
        assert len(results) == 1
        assert results[0][0] == pattern3
        assert results[0][1] == {'x': Symbol('longer')}

    def test_different_constraints_on_commutative_operation(self):
        c1 = CustomConstraint(lambda x: len(str(x)) > 1)
        c2 = CustomConstraint(lambda x: len(str(x)) == 1)
        pattern1 = fc(x_, constraint=c1)
        pattern2 = fc(x_, constraint=c2)
        pattern3 = fc(x_, b, constraint=c1)
        pattern4 = fc(x_, b, constraint=c2)
        matcher = ManyToOneMatcher(pattern1, pattern2, pattern3, pattern4)

        subject = fc(a)
        results = list(matcher.match(subject))
        assert len(results) == 1
        assert results[0][0] == pattern2
        assert results[0][1] == {'x': a}

        subject = fc(Symbol('longer'), b)
        results = sorted(matcher.match(subject))
        assert len(results) == 1
        assert results[0][0] == pattern3
        assert results[0][1] == {'x': Symbol('longer')}

        subject = fc(a, b)
        results = list(matcher.match(subject))
        assert len(results) == 1
        assert results[0][0] == pattern4
        assert results[0][1] == {'x': a}


if __name__ == '__main__':
    import matchpy.matching as tested_module
    pytest.main(['--doctest-modules', __file__, tested_module.__file__])
