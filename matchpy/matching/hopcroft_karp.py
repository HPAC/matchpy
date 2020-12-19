from collections import deque
from typing import Generic, Dict, TypeVar, Hashable, List, Tuple, Deque

THLeft = TypeVar('THLeft', bound=Hashable)
THRight = TypeVar('THRight', bound=Hashable)

FAKE_INFINITY = -1


class HopcroftKarp(Generic[THLeft, THRight]):
    """Implementation of the Hopcroft-Karp algorithm on a bipartite graph.
    The two partitions of the bipartite graph may have different types,
    which are here represented by THLeft and THRight.

    The constructor accepts a ``dict`` mapping the left vertices to the set
    of connected right vertices.

    An instance of maximum matching may be returned by
    ``.get_maximum_matching()``, while ``.get_maximum_matching_num()``
    returns both cardinality and an instance of maximum matching.

    The internal algorithm does not use sets in order to keep identical
    results across different Python versions.
    """

    def __init__(self, graph_left: Dict[THLeft, List[THRight]]):
        """Construct the HopcroftKarp class with a bipartite graph.

        Args:
            graph_left: a dictionary mapping the left-nodes to a list of
                right-nodes among which connections exist. The list shall not
                contain duplicates.

        """
        self._graph_left: Dict[THLeft, List[THRight]] = graph_left
        self._reference_distance: int = FAKE_INFINITY
        self._pair_left: Dict[THLeft, THRight] = {}
        self._pair_right: Dict[THRight, THLeft] = {}
        self._left: List[THLeft] = list(self._graph_left.keys())
        self._dist_left: Dict[THLeft, int] = {}

    def _run_hopcroft_karp(self) -> int:
        self._pair_left.clear()
        self._pair_right.clear()
        self._dist_left.clear()
        left: THLeft
        for left in self._left:
            self._dist_left[left] = FAKE_INFINITY
        matchings: int = 0
        while True:
            if not self._bfs_hopcroft_karp():
                break
            for left in self._left:
                if left in self._pair_left:
                    continue
                if self._dfs_hopcroft_karp(left):
                    matchings += 1
        return matchings

    def get_maximum_matching(self) -> Dict[THLeft, THRight]:
        """Find an instance of maximum matching for the given bipartite graph.

        Returns:
            A dictionary representing an instance of maximum matching.

        """
        matchings, maximum_matching = self.get_maximum_matching_num()
        return maximum_matching

    def get_maximum_matching_num(self) -> Tuple[int, Dict[THLeft, THRight]]:
        """Find an instance of maximum matching and the number of matchings
        found.

        Returns:
            A tuple containing the number of matchings found and a dictionary
            representing an instance of maximum matching on the given
            bipartite graph.

        """
        matchings = self._run_hopcroft_karp()
        return matchings, self._pair_left

    def _bfs_hopcroft_karp(self) -> bool:
        vertex_queue: Deque[THLeft] = deque([])
        left_vert: THLeft
        for left_vert in self._left:
            if left_vert not in self._pair_left:
                vertex_queue.append(left_vert)
                self._dist_left[left_vert] = 0
            else:
                self._dist_left[left_vert] = FAKE_INFINITY
        self._reference_distance = FAKE_INFINITY
        while True:
            if len(vertex_queue) == 0:
                break
            left_vertex: THLeft = vertex_queue.popleft()
            if self._dist_left[left_vertex] == self._reference_distance == FAKE_INFINITY:
                continue
            if self._dist_left[left_vertex] >= self._reference_distance != FAKE_INFINITY:
                continue
            right_vertex: THRight
            for right_vertex in self._graph_left[left_vertex]:
                if right_vertex not in self._pair_right:
                    if self._reference_distance == FAKE_INFINITY:
                        self._reference_distance = self._dist_left[left_vertex] + 1
                else:
                    other_left: THLeft = self._pair_right[right_vertex]
                    if self._dist_left[other_left] == FAKE_INFINITY:
                        self._dist_left[other_left] = self._dist_left[left_vertex] + 1
                        vertex_queue.append(other_left)
        return self._reference_distance != FAKE_INFINITY

    def _swap_lr(self, left: THLeft, right: THRight) -> None:
        self._pair_left[left] = right
        self._pair_right[right] = left

    def _dfs_hopcroft_karp(self, left: THLeft) -> bool:
        right: THRight
        for right in self._graph_left[left]:
            if right not in self._pair_right:
                if self._reference_distance == self._dist_left[left] + 1:
                    self._swap_lr(left, right)
                    return True
            else:
                other_left: THLeft = self._pair_right[right]
                if self._dist_left[other_left] == self._dist_left[left] + 1:
                    if self._dfs_hopcroft_karp(other_left):
                        self._swap_lr(left, right)
                        return True
        self._dist_left[left] = FAKE_INFINITY
        return False
