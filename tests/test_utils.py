# -*- coding: utf-8 -*-
import itertools
import math
import unittest

from patternmatcher.utils import fixed_sum_vector_iter


class FixedSumVectorIteratorTest(unittest.TestCase):
    limits = [(0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, math.inf), (1, math.inf), (2, math.inf)]
    max_partition_count = 3

    def test_correctness(self):
        for m in range(1, self.max_partition_count + 1):
            for limits in itertools.product(self.limits, repeat=m):
                for n in range(3 * m):
                    with self.subTest(n=n, limits=limits):
                        minVect, maxVect = zip(*limits)
                        for vect in fixed_sum_vector_iter(minVect, maxVect, n):
                            self.assertEqual(len(vect), m)
                            self.assertEqual(sum(vect), n)

                            for v, l in zip(vect, limits):
                                if v < l[0] or v > l[1]:
                                    print(vect, limits, n)
                                self.assertGreaterEqual(v, l[0])
                                self.assertLessEqual(v, l[1])

    def test_completeness(self):
        for m in range(1, self.max_partition_count + 1):
            for limits in itertools.product(self.limits, repeat=m):
                for n in range(3 * m):
                    with self.subTest(n=n, limits=limits):
                        minVect, maxVect = zip(*limits)
                        results = list(fixed_sum_vector_iter(minVect, maxVect, n))
                        realLimits = [(minimum, min(maximum, n)) for minimum, maximum in limits]
                        ranges = [list(range(minimum, maximum + 1)) for minimum, maximum in realLimits]
                        for possibleResult in itertools.product(*ranges):
                            if sum(possibleResult) != n:
                                continue
                            self.assertIn(list(possibleResult), results)

    def test_order(self):
        for m in range(1, self.max_partition_count + 1):
            for limits in itertools.product(self.limits, repeat=m):
                for n in range(3 * m):
                    with self.subTest(n=n, limits=limits):
                        minVect, maxVect = zip(*limits)
                        last_vect = [-1] * m
                        for vect in fixed_sum_vector_iter(minVect, maxVect, n):
                            self.assertGreater(vect, last_vect)
                            last_vect = vect


if __name__ == '__main__':
    unittest.main()