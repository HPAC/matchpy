# -*- coding: utf-8 -*-
import itertools
import math
import unittest

import hypothesis.strategies as st
from ddt import data, ddt, unpack
from hypothesis import example, given
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


class FixedSumVectorIteratorTest(unittest.TestCase):
    limits = [(0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, math.inf), (1, math.inf), (2, math.inf)]
    max_partition_count = 3

    def test_correctness(self):
        for m in range(self.max_partition_count + 1):
            for limits in itertools.product(self.limits, repeat=m):
                for n in range(3 * m + 1):
                    with self.subTest(n=n, limits=limits):
                        minVect, maxVect = limits and zip(*limits) or (tuple(), tuple())
                        for vect in fixed_sum_vector_iter(minVect, maxVect, n):
                            self.assertEqual(len(vect), m, 'Incorrect size of vector')
                            self.assertEqual(sum(vect), n, 'Incorrect sum of vector')

                            for v, l in zip(vect, limits):
                                self.assertGreaterEqual(v, l[0], 'Value %r out of range %r in vector %r' % (v, l, vect))
                                self.assertLessEqual(v, l[1], 'Value %r out of range %r in vector %r' % (v, l, vect))




    def test_completeness(self):
        for m in range(self.max_partition_count + 1):
            for limits in itertools.product(self.limits, repeat=m):
                for n in range(3 * m + 1):
                    with self.subTest(n=n, limits=limits):
                        minVect, maxVect = limits and zip(*limits) or (tuple(), tuple())
                        results = list(fixed_sum_vector_iter(minVect, maxVect, n))
                        self.assertTrue(is_unique_list(results), 'Got duplicate vector')

                        realLimits = [(minimum, min(maximum, n)) for minimum, maximum in limits]
                        ranges = [list(range(minimum, maximum + 1)) for minimum, maximum in realLimits]
                        for possibleResult in itertools.product(*ranges):
                            if sum(possibleResult) != n:
                                continue
                            self.assertIn(list(possibleResult), results, 'Missing expected vector %r' % (possibleResult, ))

    def test_order(self):
        for m in range(self.max_partition_count + 1):
            for limits in itertools.product(self.limits, repeat=m):
                for n in range(3 * m + 1):
                    with self.subTest(n=n, limits=limits):
                        minVect, maxVect = limits and zip(*limits) or (tuple(), tuple())
                        last_vect = [-1] * m
                        for vect in fixed_sum_vector_iter(minVect, maxVect, n):
                            self.assertGreaterEqual(vect, last_vect, 'Vectors are not in lexical order')
                            last_vect = vect


class ExtendedEuclidTest(unittest.TestCase):
    @given(st.integers(min_value=1), st.integers(min_value=1))
    def test_correctness(self, a, b):
        x, y, d = extended_euclid(a, b)
        self.assertEqual(a % d, 0)
        self.assertEqual(b % d, 0)
        self.assertEqual(a * x + b * y, d)


class BaseSolutionLinearTest(unittest.TestCase):
    @given(st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=1000), st.integers(min_value=0, max_value=1000))
    def test_correctness(self, a, b, c):
        for x, y in base_solution_linear(a, b, c):
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertEqual(a * x + b * y, c, 'Invalid solution %r,%r' % (x, y))

    @given(st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=1000), st.integers(min_value=0, max_value=1000))
    def test_completeness(self, a, b, c):
        solutions = set(base_solution_linear(a, b, c))
        for x in range(c + 1):
            for y in range(c - a * x):
                if a * x + b * y == c:
                    self.assertIn((x, y), solutions, 'Missing solution %r,%r' % (x, y))

    @given(st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=1000), st.integers(min_value=0, max_value=1000))
    def test_uniqueness(self, a, b, c):
        solutions = list(base_solution_linear(a, b, c))
        self.assertTrue(is_unique_list(solutions), 'Duplicate solution found')


class SolveLinearDiopTest(unittest.TestCase):
    @given(st.lists(st.integers(min_value=1, max_value=100), max_size=5), st.integers(min_value=0, max_value=100))
    @example([1,2,2], 4)
    def test_correctness(self, coeffs, c):
        for solution in solve_linear_diop(c, *coeffs):
            self.assertEqual(len(solution), len(coeffs), 'Solution size differs from coefficient count')
            result = sum(c*x for c, x in zip(coeffs, solution))
            for x in solution:
                self.assertGreaterEqual(x, 0)
            self.assertEqual(result, c, 'Invalid solution %r' % (solution, ))

    @given(st.lists(st.integers(min_value=1, max_value=100), max_size=5), st.integers(min_value=0, max_value=100))
    @example([1,2,2], 4)
    def test_completeness(self, coeffs, c):
        solutions = set(solve_linear_diop(c, *coeffs))
        values = [range(c // x) for x in coeffs]
        for solution2 in itertools.product(*values):
            result = sum(c*x for c, x in zip(coeffs, solution2))
            if result == c:
                self.assertIn(solution2, solutions, 'Missing solution %r' % (solution2, ))

    @given(st.lists(st.integers(min_value=1, max_value=100), max_size=5), st.integers(min_value=0, max_value=100))
    @example([1,2,2], 4)
    def test_uniqueness(self, coeffs, c):
        solutions = list(solve_linear_diop(c, *coeffs))
        self.assertTrue(is_unique_list(solutions), 'Duplicate solution found')


@st.composite
def sequence_vars(draw):
    num_vars = draw(st.integers(min_value=1, max_value=4))

    variables = []
    for i in range(num_vars):
        name = 'var%d' % i
        count = draw(st.integers(min_value=1, max_value=4))
        minimum = draw(st.integers(min_value=0, max_value=2))

        variables.append(VariableWithCount(name, count, minimum))

    return variables


@ddt
class CommutativeSequenceVariablePartitionIterTest(unittest.TestCase):
    @given(sequence_vars(), st.lists(st.integers(1, 4), min_size=1, max_size=10))
    def test_correctness_randomized(self, variables, values):
        values = Multiset(values)
        for subst in commutative_sequence_variable_partition_iter(values, variables):
            self.assertEqual(len(variables), len(subst))
            result_union = Multiset()
            for var in variables:
                self.assertGreaterEqual(len(subst[var.name]), var.minimum)
                result_union.update(subst[var.name] * var.count)
            self.assertEqual(result_union, values)

    @unpack
    @data(
        # Variables             Values      Expected iter count
        # Variables have the form (count, minimum length)
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
    )
    def test_correctness(self, variables, values, expected_iter_count):
        values = Multiset(values)
        variables = [VariableWithCount('var%d' % i, c, m) for i, (c, m) in enumerate(variables)]
        count = 0
        for subst in commutative_sequence_variable_partition_iter(values, variables):
            self.assertEqual(len(variables), len(subst), "Wrong number of variables in the substitution")
            result_union = Multiset()
            for var in variables:
                self.assertGreaterEqual(len(subst[var.name]), var.minimum, "Variable did not get its minimum number of expressions")
                result_union.update(subst[var.name] * var.count)
            self.assertEqual(result_union, values, "Substitution is not a partition of the values")
            count += 1
        self.assertEqual(count, expected_iter_count, "Invalid number of substitution in the iterable")

if __name__ == '__main__':
    unittest.main()
