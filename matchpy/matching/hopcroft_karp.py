from typing import Generic, Dict, Set, TypeVar, Hashable, List, Tuple

TLeft = TypeVar('TLeft', bound=Hashable)
TRight = TypeVar('TRight', bound=Hashable)

INT_MAX = 10000000000000


class HopcroftKarp(Generic[TLeft, TRight]):
    """
    Implementation of the Hopcroft-Karp algorithm on a bipartite graph.

    The bipartite graph has types TLeft and TRight on the two partitions.

    The constructor accepts a `dict` mapping the left vertices to the set of
    connected right vertices.

    The method `.hopcroft_karp()` (internally) finds the maximum cardinality
    matching, returning its cardinality. An instance of maximum matching may be
    returned by `.get_maximum_matching()`, while `.get_maximum_matching_num()`
    returns both cardinality and an instance of maximum matching.
    """

    def __init__(self, _graph_left: Dict[TLeft, Set[TRight]]):

        def sorting_key(x):
            return tuple(ord(i) for i in str(x))

        self._pos2left: List[TLeft] = sorted(set(_graph_left), key=sorting_key)
        self._pos2right: List[TRight] = sorted({j for i in _graph_left.values() for j in i}, key=sorting_key)
        map_left2pos = {e: i for i, e in enumerate(self._pos2left)}
        map_right2pos = {e: i for i, e in enumerate(self._pos2right)}
        # Convert the graph to integers:
        self._graph_left: Dict[int, Set[int]] = {map_left2pos[k]: {map_right2pos[j] for j in v} for k, v in _graph_left.items()}
        self._reference_distance = INT_MAX
        self._pair_left: Dict[int, int] = {}
        self._pair_right: Dict[int, int] = {}
        self._left: List[int] = []
        self._dist_left: Dict[int, int] = {}
        self._get_left_indices_vector(self._graph_left)

    def hopcroft_karp(self) -> int:
        self._pair_left.clear()
        self._pair_right.clear()
        self._dist_left.clear()
        left: int
        for left in self._left:
            self._dist_left[left] = INT_MAX
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

    def get_maximum_matching(self) -> Dict[TLeft, TRight]:
        matchings, maximum_matching = self.get_maximum_matching_num()
        return maximum_matching

    def get_maximum_matching_num(self) -> Tuple[int, Dict[TLeft, TRight]]:
        matchings = self.hopcroft_karp()
        return matchings, {self._pos2left[k]: self._pos2right[v] for k, v in self._pair_left.items()}

    def _get_left_indices_vector(self, m: Dict[int, Set[int]]) -> None:
        p: int
        for p in m:
            self._left.append(p)

    def _bfs_hopcroft_karp(self) -> bool:
        vertex_queue: List[int] = []
        left_vert: int
        for left_vert in self._left:
            if left_vert not in self._pair_left:
                vertex_queue.append(left_vert)
                self._dist_left[left_vert] = 0
            else:
                self._dist_left[left_vert] = INT_MAX
        self._reference_distance = INT_MAX
        while True:
            if len(vertex_queue) == 0:
                break
            left_vertex: int = vertex_queue.pop(0)
            if self._dist_left[left_vertex] >= self._reference_distance:
                continue
            right_vertex: int
            for right_vertex in self._graph_left[left_vertex]:
                if right_vertex not in self._pair_right:
                    if self._reference_distance == INT_MAX:
                        self._reference_distance = self._dist_left[left_vertex] + 1
                else:
                    other_left: int = self._pair_right[right_vertex]
                    if self._dist_left[other_left] == INT_MAX:
                        self._dist_left[other_left] = self._dist_left[left_vertex] + 1
                        vertex_queue.append(other_left)
        return self._reference_distance < INT_MAX

    def _swap_lr(self, left: int, right: int) -> None:
        self._pair_left[left] = right
        self._pair_right[right] = left

    def _dfs_hopcroft_karp(self, left: int) -> bool:
        right: int
        for right in self._graph_left[left]:
            if right not in self._pair_right:
                if self._reference_distance == self._dist_left[left] + 1:
                    self._swap_lr(left, right)
                    return True
            else:
                other_left: int = self._pair_right[right]
                if self._dist_left[other_left] == self._dist_left[left] + 1:
                    if self._dfs_hopcroft_karp(other_left):
                        self._swap_lr(left, right)
                        return True
        self._dist_left[left] = INT_MAX
        return False
