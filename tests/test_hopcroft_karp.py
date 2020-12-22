from typing import Dict, List

from matchpy.matching.hopcroft_karp import HopcroftKarp


class TestHopcroftKarp:
    """
    Testing the implementation of the Hopcroft Karp algorithm.
    """

    def test_hopcroft_karp(self):

        graph: Dict[int, List[str]] = {
            0: ["v0", "v1"],
            1: ["v0", "v4"],
            2: ["v2", "v3"],
            3: ["v0", "v4"],
            4: ["v0", "v3"],
        }
        expected: Dict[int, str] = {0: "v1", 1: "v4", 2: "v2", 3: "v0", 4: "v3"}
        hk = HopcroftKarp[int, str](graph)
        matchings, maximum_matching = hk.get_maximum_matching_num()
        assert maximum_matching == expected
        assert matchings == 5

        graph: Dict[str, List[int]] = {'A': [1, 2], 'B': [2, 3], 'C': [2], 'D': [3, 4, 5, 6],
                                       'E': [4, 7], 'F': [7], 'G': [7]}
        expected: Dict[str, int] = {'A': 1, 'B': 3, 'C': 2, 'D': 5, 'E': 4, 'F': 7}
        hk = HopcroftKarp[str, int](graph)
        matchings, maximum_matching = hk.get_maximum_matching_num()
        assert maximum_matching == expected
        assert matchings == 6

        graph: Dict[int, List[str]] = {1: ['a', 'c'], 2: ['a', 'c'], 3: ['c', 'b'], 4: ['e']}
        expected: Dict[int, str] = {1: 'a', 2: 'c', 3: 'b', 4: 'e'}
        hk = HopcroftKarp[int, str](graph)
        matchings, maximum_matching = hk.get_maximum_matching_num()
        assert maximum_matching == expected
        assert matchings == 4

        graph: Dict[str, List[int]] = {'A': [3, 4], 'B': [3, 4], 'C': [3], 'D': [1, 5, 7],
                                       'E': [1, 2, 7], 'F': [2, 8], 'G': [6], 'H': [2, 4, 8]}
        expected: Dict[str, int] = {'A': 3, 'B': 4, 'D': 1, 'E': 7, 'F': 8, 'G': 6, 'H': 2}
        hk = HopcroftKarp[str, int](graph)
        matchings, maximum_matching = hk.get_maximum_matching_num()
        assert maximum_matching == expected
        assert matchings == 7
