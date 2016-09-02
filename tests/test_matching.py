# -*- coding: utf-8 -*-
import unittest
import itertools
import math

import hypothesis.strategies as st
from ddt import data, ddt, unpack
from hypothesis import given, example

from patternmatcher.matching import BipartiteGraph, find_cycle, _DGM, enum_maximum_matchings_iter

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
class RandomizedBipartiteMatchTest(unittest.TestCase):    
    @given(bipartite_graph())
    @example(BipartiteGraph(map(lambda x: (x, True), itertools.product(range(3), repeat=2))))
    def test_correctness(self, graph):
        matching = graph.find_matching()
        size = len(matching)
        matchings = {frozenset(matching.items())}
        DGM = _DGM(graph, matching)
        for matching in enum_maximum_matchings_iter(graph, matching, DGM):
            self.assertEqual(len(matching), size, 'Matching has a different size than the first one')
            for kv in matching.items():
                self.assertIn(kv, graph, 'Matching contains an edge that was not in the graph')
            frozen_matching = frozenset(matching.items())
            self.assertNotIn(frozen_matching, matchings, "Matching was duplicate")
    
    @unpack
    @data(*filter(lambda x: x[0] >= x[1], itertools.product(range(1, 8), range(1, 8))))
    def test_completeness(self, n, m):
        graph = BipartiteGraph(map(lambda x: (x, True), itertools.product(range(n), range(m))))
        matching = graph.find_matching()
        DGM = _DGM(graph, matching)
        count = len(list(enum_maximum_matchings_iter(graph, matching, DGM)))
        expected_count = math.factorial(n) / math.factorial(n - m) - 1
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