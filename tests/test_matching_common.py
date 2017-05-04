# -*- coding: utf-8 -*-
import pytest

from matchpy.expressions.constraints import CustomConstraint
from matchpy.expressions import Wildcard, Symbol
from matchpy.matching._common import CommutativePatternsParts
from .common import *

constr1 = CustomConstraint(lambda x, y: x == y)
constr2 = CustomConstraint(lambda x, y: x != y)


class TestCommutativePatternsParts:
    @pytest.mark.parametrize(
        '   expressions,                    constant,           sequence_variables,                 fixed_variables,            fixed_terms',
        [
            ([],                            [],                 [],                                 [],                         []),
            ([a],                           [a],                [],                                 [],                         []),
            ([a, b],                        [a, b],             [],                                 [],                         []),
            ([x_],                          [],                 [],                                 [('x', None)],              []),
            ([x_, y_],                      [],                 [],                                 [('x', None), ('y', None)], []),
            ([f(a)],                        [f(a)],             [],                                 [],                         []),
            ([f(x__)],                      [],                 [],                                 [],                         [f(x__)]),
            ([f(a), f(b)],                  [f(a), f(b)],       [],                                 [],                         []),
            ([x__],                         [],                 [('x', 1, False)],                  [],                         []),
            ([x___],                        [],                 [('x', 0, False)],                  [],                         []),
            ([x__, y___],                   [],                 [('x', 1, False), ('y', 0, False)], [],                         []),
            ([f_c(x_)],                     [],                 [],                                 [],                         [f_c(x_)]),
            ([f_c(x_, a)],                  [],                 [],                                 [],                         [f_c(x_, a)]),
            ([f_c(x_, a), f_c(x_, b)],      [],                 [],                                 [],                         [f_c(x_, a), f_c(x_, b)]),
            ([f_c(a)],                      [f_c(a)],           [],                                 [],                         []),
            ([f_c(a), f_c(b)],              [f_c(a), f_c(b)],   [],                                 [],                         []),
            ([a, x_, x__, f(x_), f_c(x_)],  [a],                [('x', 1, False)],                  [('x', None)],              [f(x_), f_c(x_)]),
            ([__],                          [],                 [],                                 [],                         []),
            ([_],                           [],                 [],                                 [],                         []),
            ([_, _],                        [],                 [],                                 [],                         []),
            ([___],                         [],                 [],                                 [],                         []),
            ([___, __, _],                  [],                 [],                                 [],                         []),
            ([__, x_],                      [],                 [],                                 [('x', None)],              []),
            ([__, x__],                     [],                 [('x', 1, False)],                  [],                         []),
            ([s_],                          [],                 [],                                 [('s', Symbol)],            []),
        ]
    )  # yapf: disable
    def test_parts(self, expressions, constant, sequence_variables, fixed_variables, fixed_terms):
        parts = CommutativePatternsParts(None, *expressions)

        assert constant == sorted(parts.constant)

        assert len(sequence_variables) == len(parts.sequence_variables)
        for name, min_count, wrap in sequence_variables:
            assert name in parts.sequence_variables
            assert name in parts.sequence_variables_min
            assert name in parts.sequence_variables_wrap
            assert min_count == parts.sequence_variables_min[name]
            assert wrap == parts.sequence_variables_wrap[name]

        assert len(fixed_variables) == len(parts.fixed_variables)
        for name, stype in fixed_variables:
            assert name in parts.fixed_variables
            assert name in parts.fixed_variables_type
            assert stype == parts.fixed_variables_type[name]

        assert fixed_terms == sorted(parts.fixed_terms)

        assert sum(c for _, c, _ in sequence_variables) == parts.sequence_variable_min_length
        assert len(fixed_variables) == parts.fixed_variable_length

        if any(isinstance(o, Wildcard) and not o.variable_name for o in expressions):
            fixed = all(wc.fixed_size for wc in expressions if isinstance(wc, Wildcard) and not wc.variable_name)
            length = sum(wc.min_count for wc in expressions if isinstance(wc, Wildcard) and not wc.variable_name)

            assert parts.wildcard_fixed is fixed
            assert parts.wildcard_min_length == length
        else:
            assert parts.wildcard_fixed is None
