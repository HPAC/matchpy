from typing import Generic, Dict, Set, TypeVar, Hashable, List, Tuple

TLeft = TypeVar('TLeft', bound=Hashable)
TRight = TypeVar('TRight', bound=Hashable)

INT_MAX = 10000000000000


class HopcroftKarp(Generic[TLeft, TRight]):
    """
    Implementation of the Hopcroft-Karp algorithm on a bipartite graph.

    The bipartite graph has types TLeft and TRight on the two partitions.

    The constructor accepts a `map` mapping the left vertices to the set of
    connected right vertices.

    The method `.hopcroft_karp()` finds the maximum cardinality matching,
    returning its cardinality. The matching will be stored in the file
    `pair_left`
    and `pair_right` after the matching is found.
    """

    def __init__(self, _graph_left: Dict[TLeft, Set[TRight]]):
        self._graph_left: Dict[TLeft, Set[TRight]] = _graph_left
        self._reference_distance = INT_MAX
        self.pair_left: Dict[TLeft, TRight] = {}
        self.pair_right: Dict[TRight, TLeft] = {}
        self._left: List[TLeft] = []
        self._dist_left: Dict[TLeft, int] = {}
        self._get_left_indices_vector(_graph_left)

    def hopcroft_karp(self) -> int:
        self.pair_left.clear()
        self.pair_right.clear()
        self._dist_left.clear()
        left: TLeft
        for left in self._left:
            self._dist_left[left] = INT_MAX
        matchings: int = 0
        while True:
            if not self._bfs_hopcroft_karp():
                break
            left: TLeft
            for left in self._left:
                if left in self.pair_left:
                    continue
                if self._dfs_hopcroft_karp(left):
                    matchings += 1
        return matchings

    def get_maximum_matching(self) -> Dict[TLeft, TRight]:
        self.hopcroft_karp()
        return self.pair_left

    def _get_left_indices_vector(self, m: Dict[TLeft, Set[TRight]]) -> None:
        p: Tuple[TLeft, Set[TRight]]
        for p in m.items():
            self._left.append(p[0])

    def _bfs_hopcroft_karp(self) -> bool:
        vertex_queue: List[TLeft] = []
        left_vert: TLeft
        for left_vert in self._left:
            if left_vert not in self.pair_left:
                vertex_queue.append(left_vert)
                self._dist_left[left_vert] = 0
            else:
                self._dist_left[left_vert] = INT_MAX
        self._reference_distance = INT_MAX
        while True:
            if len(vertex_queue) == 0:
                break
            left_vertex: TLeft = vertex_queue.pop(0)
            if self._dist_left[left_vertex] >= self._reference_distance:
                continue
            right_vertex: TRight
            for right_vertex in self._graph_left[left_vertex]:
                if right_vertex not in self.pair_right:
                    if self._reference_distance == INT_MAX:
                        self._reference_distance = self._dist_left[left_vertex] + 1
                else:
                    other_left: TLeft = self.pair_right[right_vertex]
                    if self._dist_left[other_left] == INT_MAX:
                        self._dist_left[other_left] = self._dist_left[left_vertex] + 1
                        vertex_queue.append(other_left)
        return self._reference_distance < INT_MAX

    def _swap_lr(self, left: TLeft, right: TRight) -> None:
        self.pair_left[left] = right
        self.pair_right[right] = left

    def _dfs_hopcroft_karp(self, left: TLeft) -> bool:
        right: TRight
        for right in self._graph_left[left]:
            if right not in self.pair_right:
                if self._reference_distance == self._dist_left[left] + 1:
                    self._swap_lr(left, right)
                    return True
            else:
                other_left: TLeft = self.pair_right[right]
                if self._dist_left[other_left] == self._dist_left[left] + 1:
                    if self._dfs_hopcroft_karp(other_left):
                        self._swap_lr(left, right)
                        return True
        self._dist_left[left] = INT_MAX
        return False
