# -*- coding: utf-8 -*-
import unittest

import hypothesis.strategies as st
from ddt import data, ddt, unpack
from hypothesis import assume, given

from patternmatcher.matching import BipartiteGraph, find_cycle

@st.composite
def bipartite_graph(draw):
    n = draw(st.integers(min_value=1, max_value=8))
    m = draw(st.integers(min_value=n, max_value=10))

    graph = BipartiteGraph()
    for i in range(n):
        for j in range(m):
            b = draw(st.booleans())
            if b:
                graph[i,j] = b

    return graph

@unittest.skip('temp')
class RandomizedBipartiteMatchTest(unittest.TestCase):
    i = 0
    
    @given(bipartite_graph())
    def test_find_cycle_correct(self, graph):
        print(graph)
        graph.as_graph().render('G%d' % self.i)
        self.i += 1

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