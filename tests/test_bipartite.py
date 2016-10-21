# -*- coding: utf-8 -*-
import doctest
import itertools
import math
import unittest

from ddt import data, ddt, unpack
import hypothesis.strategies as st
from hypothesis import given

import patternmatcher.bipartite as bipartite
from patternmatcher.bipartite import (BipartiteGraph, _DirectedMatchGraph,
                                      enum_maximum_matchings_iter)


@st.composite
def bipartite_graph(draw):
    m = draw(st.integers(min_value=1, max_value=4))
    n = draw(st.integers(min_value=m, max_value=5))

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
        matchings = set()
        for matching in enum_maximum_matchings_iter(graph):
            if size is None:
                size = len(matching)
            self.assertEqual(len(matching), size, 'Matching has a different size than the first one')
            for kv in matching.items():
                self.assertIn(kv, graph, 'Matching contains an edge that was not in the graph')
            frozen_matching = frozenset(matching.items())
            self.assertNotIn(frozen_matching, matchings, "Matching was duplicate")
            matchings.add(frozen_matching)

    @unpack
    @data(*filter(lambda x: x[0] >= x[1], itertools.product(range(1, 6), range(0, 4))))
    def test_completeness(self, n, m):
        graph = BipartiteGraph(map(lambda x: (x, True), itertools.product(range(n), range(m))))
        count = sum(1 for _ in enum_maximum_matchings_iter(graph))
        expected_count = m > 0 and math.factorial(n) / math.factorial(n - m) or 0
        self.assertEqual(count, expected_count)



@ddt
class DirectedMatchGraphFindCycleTest(unittest.TestCase):
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
        dmg = _DirectedMatchGraph({}, {})
        dmg.update(graph)
        cycle = dmg.find_cycle()
        if len(expected_cycle) > 0:
            self.assertIn(expected_cycle[0], cycle)
            start = cycle.index(expected_cycle[0])
            cycle = cycle[start:] + cycle[:start]
        self.assertListEqual(cycle, expected_cycle)


class BipartiteGraphTest(unittest.TestCase):
    def test_setitem(self):
        graph = BipartiteGraph()

        graph[0,1] = True

        with self.assertRaises(TypeError):
            graph[0] = True

        with self.assertRaises(TypeError):
            graph[0,] = True

        with self.assertRaises(TypeError):
            graph[0,1,2] = True

    def test_getitem(self):
        graph = BipartiteGraph({(0,0): True})

        self.assertEqual(graph[0,0], True)

        with self.assertRaises(TypeError):
            _ = graph[0]

        with self.assertRaises(TypeError):
            _ = graph[0,]

        with self.assertRaises(TypeError):
            _ = graph[0,1,2]

        with self.assertRaises(KeyError):
            _ = graph[0,1]

    def test_delitem(self):
        graph = BipartiteGraph({(0,0): True})

        self.assertIn((0,0), graph)

        del graph[0,0]

        self.assertNotIn((0,0), graph)

        with self.assertRaises(TypeError):
            del graph[0]

        with self.assertRaises(TypeError):
            del graph[0,]

        with self.assertRaises(TypeError):
            del graph[0,1,2]

        with self.assertRaises(KeyError):
            del graph[0,1]

    def test_limited_to(self):
        graph = BipartiteGraph({(0, 0): True, (1, 0): True, (1, 1): True, (0, 1): True})

        self.assertEqual(graph.limited_to({0}, {0}), {(0, 0): True})
        self.assertEqual(graph.limited_to({0, 1}, {1}), {(0, 1): True, (1, 1): True})
        self.assertEqual(graph.limited_to({1}, {1}), {(1, 1): True})
        self.assertEqual(graph.limited_to({1}, {0, 1}), {(1, 0): True, (1, 1): True})
        self.assertEqual(graph.limited_to({0, 1}, {0, 1}), graph)


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(bipartite))
    return tests

if __name__ == '__main__':
    unittest.main()
