# -*- coding: utf-8 -*-
import itertools
import math
import unittest

import hypothesis.strategies as st
from ddt import data, ddt, unpack
from hypothesis import given

from patternmatcher.matching import (BipartiteGraph,
                                     enum_maximum_matchings_iter, find_cycle)


@st.composite
def bipartite_graph(draw):
    m = draw(st.integers(min_value=1, max_value=8))
    n = draw(st.integers(min_value=m, max_value=10))

    graph = BipartiteGraph()
    for i in range(n):
        for j in range(m):
            b = draw(st.booleans())
            if b:
                graph[i,j] = b

    return graph

@ddt
class EnumMaximumMatchingsIterTest(unittest.TestCase):
    @given(bipartite_graph())
    def test_correctness(self, graph):
        size = None
        matchings = {}
        for matching in enum_maximum_matchings_iter(graph):
            if size is None:
                size = len(matching)
            self.assertEqual(len(matching), size, 'Matching has a different size than the first one')
            for kv in matching.items():
                self.assertIn(kv, graph, 'Matching contains an edge that was not in the graph')
            frozen_matching = frozenset(matching.items())
            self.assertNotIn(frozen_matching, matchings, "Matching was duplicate")

    @unpack
    @data(*filter(lambda x: x[0] >= x[1], itertools.product(range(1, 6), range(0, 4))))
    def test_completeness(self, n, m):
        graph = BipartiteGraph(map(lambda x: (x, True), itertools.product(range(n), range(m))))
        count = sum(1 for _ in enum_maximum_matchings_iter(graph))
        expected_count = m > 0 and math.factorial(n) / math.factorial(n - m) or 0
        self.assertEqual(count, expected_count)



@ddt
class FindCycleTest(unittest.TestCase):
    @unpack
    @data(
        ({},                        []),
        ({0: {1}},                  []),
        ({0: {1}, 1: {2}},          []),
        ({0: {1}, 1: {0}},          [0, 1]),
        ({0: {1}, 1: {0}},          [1, 0]),
        ({0: {1}, 1: {0, 2}},       [0, 1]),
        ({0: {1, 2}, 1: {0, 2}},    [0, 1]),
        ({0: {1, 2}, 1: {0}},       [0, 1]),
        ({0: {1}, 1: {2}, 2: {0}},  [0, 1, 2]),
        ({0: {2}, 1: {2}},          []),
        ({0: {2}, 1: {2}, 2: {0}},  [0, 2]),
        ({0: {2}, 1: {2}, 2: {1}},  [1, 2]),
    )
    def test_find_cycle(self, graph, expected_cycle):
        cycle = find_cycle(graph)
        if len(expected_cycle) > 0:
            self.assertIn(expected_cycle[0], cycle)
            start = cycle.index(expected_cycle[0])
            cycle = cycle[start:] + cycle[:start]
        self.assertListEqual(cycle, expected_cycle)


if __name__ == '__main__':
    unittest.main()
