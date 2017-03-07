# -*- coding: utf-8 -*-
import pytest

from matchpy.expressions.constraints import CustomConstraint
from matchpy.expressions.expressions import Wildcard
from matchpy.matching._common import CommutativePatternsParts
from .common import *

constr1 = CustomConstraint(lambda x, y: x == y)
constr2 = CustomConstraint(lambda x, y: x != y)


class TestCommutativePatternsParts:
    @pytest.mark.parametrize(
        '   expressions,                    constant,           syntactic,      sequence_variables,     fixed_variables,        rest',
        [
            ([],                            [],                 [],             [],                     [],                     []),
            ([a],                           [a],                [],             [],                     [],                     []),
            ([a, b],                        [a, b],             [],             [],                     [],                     []),
            ([x_],                          [],                 [],             [],                     [('x', 1)],             []),
            ([x_, y_],                      [],                 [],             [],                     [('x', 1), ('y', 1)],   []),
          # ([x2],                          [],                 [],             [],                     [('x', 2)],             []),
            ([f(x_)],                       [],                 [f(x_)],        [],                     [],                     []),
            ([f(x_), f(y_)],                [],                 [f(x_), f(y_)], [],                     [],                     []),
            ([f(a)],                        [f(a)],             [],             [],                     [],                     []),
            ([f(x__)],                      [],                 [],             [],                     [],                     [f(x__)]),
            ([f(a), f(b)],                  [f(a), f(b)],       [],             [],                     [],                     []),
            ([x__],                         [],                 [],             [('x', 1)],             [],                     []),
            ([x___],                        [],                 [],             [('x', 0)],             [],                     []),
            ([x__, y___],                   [],                 [],             [('x', 1), ('y', 0)],   [],                     []),
            ([f_c(x_)],                     [],                 [],             [],                     [],                     [f_c(x_)]),
            ([f_c(x_, a)],                  [],                 [],             [],                     [],                     [f_c(x_, a)]),
            ([f_c(x_, a), f_c(x_, b)],      [],                 [],             [],                     [],                     [f_c(x_, a), f_c(x_, b)]),
            ([f_c(a)],                      [f_c(a)],           [],             [],                     [],                     []),
            ([f_c(a), f_c(b)],              [f_c(a), f_c(b)],   [],             [],                     [],                     []),
            ([a, x_, x__, f(x_), f_c(x_)],  [a],                [f(x_)],        [('x', 1)],             [('x', 1)],             [f_c(x_)]),
            ([__],                          [],                 [],             [],                     [],                     []),
            ([_],                           [],                 [],             [],                     [],                     []),
            ([_, _],                        [],                 [],             [],                     [],                     []),
            ([___],                         [],                 [],             [],                     [],                     []),
            ([___, __, _],                  [],                 [],             [],                     [],                     []),
            ([__, x_],                      [],                 [],             [],                     [('x', 1)],             []),
            ([__, x__],                     [],                 [],             [('x', 1)],             [],                     []),
        ]
    )  # yapf: disable
    def test_parts(self, expressions, constant, syntactic, sequence_variables, fixed_variables, rest):
        parts = CommutativePatternsParts(None, *expressions)

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

        if any(isinstance(o, Wildcard) and not o.variable_name for o in expressions):
            fixed = all(wc.fixed_size for wc in expressions if isinstance(wc, Wildcard) and not wc.variable_name)
            length = sum(wc.min_count for wc in expressions if isinstance(wc, Wildcard) and not wc.variable_name)

            assert parts.wildcard_fixed is fixed
            assert parts.wildcard_min_length == length
        else:
            assert parts.wildcard_fixed is None
