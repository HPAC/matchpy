# -*- coding: utf-8 -*-
import itertools
import math
import unittest

import hypothesis.strategies as st
from hypothesis import given, example

from patternmatcher.utils import extended_euclid, fixed_sum_vector_iter, base_solution_linear, solve_linear_diop

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


if __name__ == '__main__':
    unittest.main()
