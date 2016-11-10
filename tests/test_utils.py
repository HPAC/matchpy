# -*- coding: utf-8 -*-
import itertools
import math

import hypothesis.strategies as st
from hypothesis import example, given, assume
import pytest
from multiset import Multiset

from patternmatcher.utils import (VariableWithCount, base_solution_linear,
                                  commutative_sequence_variable_partition_iter,
                                  extended_euclid, fixed_sum_vector_iter,
                                  solve_linear_diop)


def is_unique_list(l):
    for i, v1 in enumerate(l):
        for v2 in l[:i]:
            if v1 == v2:
                return False
    return True


class TestFixedSumVectorIterator:
    limits = [(0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, math.inf), (1, math.inf), (2, math.inf)]
    max_partition_count = 3

    def test_correctness(self):
        for m in range(self.max_partition_count + 1):
            for limits in itertools.product(self.limits, repeat=m):
                for n in range(3 * m + 1):
                    minVect, maxVect = limits and zip(*limits) or (tuple(), tuple())
                    for vect in fixed_sum_vector_iter(minVect, maxVect, n):
                        assert len(vect) == m, "Incorrect size of vector"
                        assert sum(vect) == n, "Incorrect sum of vector"

                        for v, l in zip(vect, limits):
                            assert v >= l[0], "Value {!r} out of range {!r} in vector {!r}".format(v, l, vect)
                            assert v <= l[1], "Value {!r} out of range {!r} in vector {!r}".format(v, l, vect)




    def test_completeness(self):
        for m in range(self.max_partition_count + 1):
            for limits in itertools.product(self.limits, repeat=m):
                for n in range(3 * m + 1):
                    minVect, maxVect = limits and zip(*limits) or (tuple(), tuple())
                    results = list(fixed_sum_vector_iter(minVect, maxVect, n))
                    assert is_unique_list(results), "Got duplicate vector"

                    realLimits = [(minimum, min(maximum, n)) for minimum, maximum in limits]
                    ranges = [list(range(minimum, maximum + 1)) for minimum, maximum in realLimits]
                    for possibleResult in itertools.product(*ranges):
                        if sum(possibleResult) != n:
                            continue
                        assert list(possibleResult) in results, "Missing expected vector {!r}".format(possibleResult)

    def test_order(self):
        for m in range(self.max_partition_count + 1):
            for limits in itertools.product(self.limits, repeat=m):
                for n in range(3 * m + 1):
                    minVect, maxVect = limits and zip(*limits) or (tuple(), tuple())
                    last_vect = [-1] * m
                    for vect in fixed_sum_vector_iter(minVect, maxVect, n):
                        assert vect >= last_vect, "Vectors are not in lexical order"
                        last_vect = vect


@given(st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=1000))
def test_extended_euclid(a, b):
    x, y, d = extended_euclid(a, b)
    assert a % d == 0
    assert b % d == 0
    assert a * x + b * y == d


class TestBaseSolutionLinear:
    @given(st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=1000), st.integers(min_value=0, max_value=1000))
    def test_correctness(self, a, b, c):
        for x, y in base_solution_linear(a, b, c):
            assert x >= 0
            assert y >= 0
            assert a * x + b * y == c, "Invalid solution {!r},{!r}".format(x, y)

    @given(st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=1000), st.integers(min_value=0, max_value=1000))
    def test_completeness(self, a, b, c):
        solutions = set(base_solution_linear(a, b, c))
        for x in range(c + 1):
            for y in range(c - a * x):
                if a * x + b * y == c:
                    assert (x, y) in solutions, "Missing solution {!r},{!r}".format(x, y)

    @given(st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=1000), st.integers(min_value=0, max_value=1000))
    def test_uniqueness(self, a, b, c):
        solutions = list(base_solution_linear(a, b, c))
        assert is_unique_list(solutions), "Duplicate solution found"


class TestSolveLinearDiop:
    @staticmethod
    def _limit_possible_solution_count(coeffs, c):
        total_solutions_approx = 1
        for coeff in coeffs:
            if c % coeff == 0:
                total_solutions_approx *= c / coeff
        assume(total_solutions_approx <= 100)

    @given(st.lists(st.integers(min_value=1, max_value=100), max_size=5), st.integers(min_value=0, max_value=100))
    @example([1,2,2], 4)
    def test_correctness(self, coeffs, c):
        self._limit_possible_solution_count(coeffs, c)
        for solution in solve_linear_diop(c, *coeffs):
            assert len(solution) == len(coeffs), "Solution size differs from coefficient count"
            result = sum(c*x for c, x in zip(coeffs, solution))
            for x in solution:
                assert x >= 0
            assert result == c, "Invalid solution {!r}".format(solution)

    @given(st.lists(st.integers(min_value=1, max_value=100), max_size=5), st.integers(min_value=0, max_value=100))
    @example([1,2,2], 4)
    def test_completeness(self, coeffs, c):
        self._limit_possible_solution_count(coeffs, c)
        solutions = set(solve_linear_diop(c, *coeffs))
        values = [range(c // x) for x in coeffs]
        for solution2 in itertools.product(*values):
            result = sum(c*x for c, x in zip(coeffs, solution2))
            if result == c:
                assert solution2 in solutions, "Missing solution {!r}".format(solution2)

    @given(st.lists(st.integers(min_value=1, max_value=100), max_size=5), st.integers(min_value=0, max_value=100))
    @example([1,2,2], 4)
    def test_uniqueness(self, coeffs, c):
        self._limit_possible_solution_count(coeffs, c)
        solutions = list(solve_linear_diop(c, *coeffs))
        assert is_unique_list(solutions), "Duplicate solution found"


@st.composite
def sequence_vars(draw):
    num_vars = draw(st.integers(min_value=1, max_value=4))

    variables = []
    for i in range(num_vars):
        name = 'var{:d}'.format(i)
        count = draw(st.integers(min_value=1, max_value=4))
        minimum = draw(st.integers(min_value=0, max_value=2))

        variables.append(VariableWithCount(name, count, minimum))

    return variables


class TestCommutativeSequenceVariablePartitionIter:
    @given(sequence_vars(), st.lists(st.integers(1, 4), min_size=1, max_size=10))
    def test_correctness_randomized(self, variables, values):
        values = Multiset(values)
        for subst in commutative_sequence_variable_partition_iter(values, variables):
            assert len(variables) == len(subst)
            result_union = Multiset()
            for var in variables:
                assert len(subst[var.name]) >= var.minimum
                result_union.update(subst[var.name] * var.count)
            assert result_union == values

    @pytest.mark.parametrize(
        '   variables,              values,     expected_iter_count',
        #   Variables have the form (count, minimum length)
        [
            ([],                    'a',        0),
            ([],                    '',         1),
            ([(1, 0)],              '',         1),
            ([(1, 0)],              'a',        1),
            ([(1, 1)],              '',         0),
            ([(1, 1)],              'a',        1),
            ([(1, 2)],              'a',        0),
            ([(2, 0)],              '',         1),
            ([(2, 0)],              'a',        0),
            ([(2, 1)],              '',         0),
            ([(2, 1)],              'a',        0),
            ([(2, 2)],              'a',        0),
            ([(2, 0)],              'ab',       0),
            ([(2, 1)],              'ab',       0),
            ([(2, 2)],              'ab',       0),
            ([(2, 0)],              'aa',       1),
            ([(2, 1)],              'aa',       1),
            ([(2, 2)],              'aa',       0),
            ([(2, 0)],              'aaa',      0),
            ([(2, 1)],              'aaa',      0),
            ([(2, 2)],              'aaa',      0),
            ([(2, 0)],              'aabb',     1),
            ([(2, 1)],              'aabb',     1),
            ([(2, 2)],              'aabb',     1),
            ([(1, 0), (1, 0)],      '',         1),
            ([(1, 0), (1, 0)],      'a',        2),
            ([(1, 1), (1, 0)],      '',         0),
            ([(1, 1), (1, 0)],      'a',        1),
            ([(1, 0), (1, 0)],      'aa',       3),
            ([(1, 1), (1, 0)],      'aa',       2),
            ([(1, 0), (1, 0)],      'ab',       4),
            ([(1, 1), (1, 0)],      'ab',       3),
            ([(1, 0), (1, 0)],      'aaa',      4),
            ([(1, 1), (1, 0)],      'aaa',      3),
            ([(1, 0), (1, 0)],      'aab',      6),
            ([(1, 1), (1, 0)],      'aab',      5),
            ([(1, 0), (1, 0)],      'a',        2),
            ([(1, 1), (1, 0)],      'a',        1),
            ([(2, 0), (1, 0)],      '',         1),
            ([(2, 0), (1, 0)],      'aa',       2),
            ([(2, 1), (1, 0)],      '',         0),
            ([(2, 1), (1, 0)],      'aa',       1),
            ([(2, 0), (1, 0)],      'ab',       1),
            ([(2, 1), (1, 0)],      'ab',       0),
            ([(2, 0), (1, 0)],      'aaa',      2),
            ([(2, 1), (1, 0)],      'aaa',      1),
            ([(2, 0), (1, 0)],      'aab',      2),
            ([(2, 1), (1, 0)],      'aab',      1),
        ]
    )
    def test_correctness(self, variables, values, expected_iter_count):
        values = Multiset(values)
        variables = [VariableWithCount('var{:d}'.format(i), c, m) for i, (c, m) in enumerate(variables)]
        count = 0
        for subst in commutative_sequence_variable_partition_iter(values, variables):
            assert len(variables) == len(subst), "Wrong number of variables in the substitution"
            result_union = Multiset()
            for var in variables:
                assert len(subst[var.name]) >= var.minimum, "Variable did not get its minimum number of expressions"
                result_union.update(subst[var.name] * var.count)
            assert result_union == values, "Substitution is not a partition of the values"
            count += 1
        assert count == expected_iter_count, "Invalid number of substitution in the iterable"

if __name__ == '__main__':
    import patternmatcher.utils as tested_module
    pytest.main(['--doctest-modules', __file__, tested_module.__file__])
