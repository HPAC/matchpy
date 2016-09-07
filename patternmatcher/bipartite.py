# -*- coding: utf-8 -*-

from typing import Dict, List, Set, Tuple, TypeVar, Union, Iterator, Generic, cast, Hashable

from graphviz import Digraph, Graph
from hopcroftkarp import HopcroftKarp

T = TypeVar('T')
TLeft = TypeVar('TLeft', bound=Hashable)
TRight = TypeVar('TRight', bound=Hashable)
TEdge = TypeVar('TEdge')

LEFT = 0
RIGHT = 1

class BipartiteGraph(Dict[Tuple[TLeft, TRight], TEdge], Generic[TLeft, TRight, TEdge]):
    """A bipartite graph representation.

    This class is a specilized dictionary, where each edge is represented by a 2-tuple that is used as a key in the
    dictionary. The value can either be `True` or any value that you want to associate with the edge.
    """

    def __setitem__(self, key, value):
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError('Key must a 2-tuple')
        super(BipartiteGraph, self).__setitem__(key, value)

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError('Key must a 2-tuple')
        return super(BipartiteGraph, self).__getitem__(key)

    def __delitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError('Key must a 2-tuple')
        return super(BipartiteGraph, self).__delitem__(key)

    def as_graph(self) -> Graph: # pragma: no cover
        """Returns a :class:`graphviz.Graph` representation of this bipartite graph."""
        graph = Graph()

        nodes_left = {} # type: Dict[TLeft,str]
        nodes_right = {} # type: Dict[TRight,str]
        node_id = 0
        for (left, right), value in self.items():
            if left not in nodes_left:
                name = 'node%d' % node_id
                nodes_left[left] = name
                graph.node(name, label=str(left))
                node_id += 1
            if right not in nodes_right:
                name = 'node%d' % node_id
                nodes_right[right] = name
                graph.node(name, label=str(right))
                node_id += 1
            edge_label = value is not True and str(value) or ''
            graph.edge(nodes_left[left], nodes_right[right], edge_label)

        return graph

    def find_matching(self) -> Dict[TLeft,TRight]:
        """Finds a matching in the bipartite graph.

        This is done using the Hopcroft-Karp algorithm with an implementation from the
        `hopcroftkarp` package.

        :returns: A dictionary where each edge of the matching is represeted by a key-value pair
        with the key being from the left part of the graph and the value from te right part.
        """
        # The directed graph is represented as a dictionary of edges
        # The key is the tail of all edges which are represented by the value
        # The value is a set of heads for the all edges originating from the tail (key)
        # In addition, the graph stores which part of the bipartite graph a node originated from
        # to avoid problems when a value exists in both halfs.
        # Only one direction of the undirected edge is needed for the HopcroftKarp class
        directed_graph = {} # type: Dict[Tuple[int, TLeft],Set[Tuple[int,TRight]]]

        for (left, right) in self:
            tail = (LEFT, left)
            head = (RIGHT, right)
            if tail not in directed_graph:
                directed_graph[tail] = {head}
            else:
                directed_graph[tail].add(head)

        matching = HopcroftKarp(directed_graph).maximum_matching()

        # Filter out the partitions (LEFT and RIGHT) and only return the matching edges
        # that go from LEFT to RIGHT
        return dict((tail[1], head[1]) for tail, head in matching.items() if tail[0] == LEFT)

    def without_nodes(self, edge: Tuple[TLeft, TRight]) -> 'BipartiteGraph[TLeft, TRight, TEdge]':
        """Returns a copy of this bipartite graph with the given edge and its adjacent nodes removed."""
        return BipartiteGraph(((n1, n2), v) for (n1, n2), v in self.items() if n1 != edge[0] and n2 != edge[1])

    def without_edge(self, edge: Tuple[TLeft, TRight]) -> 'BipartiteGraph[TLeft, TRight, TEdge]':
        """Returns a copy of this bipartite graph with the given edge removed."""
        return BipartiteGraph((e2, v) for e2, v in self.items() if edge != e2)


class _DirectedMatchGraph(Dict[Tuple[int,Union[TLeft,TRight]],Set[Tuple[int,Union[TLeft,TRight]]]], Generic[TLeft, TRight]):
    def __init__(self, graph: BipartiteGraph[TLeft, TRight, TEdge], matching: Dict[TLeft, TRight]) -> None:
        super(_DirectedMatchGraph, self).__init__()
        for (tail, head) in graph:
            if tail in matching and matching[tail] == head:
                self[(LEFT, tail)] = {(RIGHT, head)}
            else:
                if (RIGHT, head) not in self:
                    self[(RIGHT, head)] = set()
                self[(RIGHT, head)].add((LEFT, tail))

    def as_graph(self) -> Digraph: # pragma: no cover
        """Returns a :class:`graphviz.Digraph` representation of this directed match graph."""
        graph = Digraph()

        subgraphs = [Digraph(graph_attr={'rank': 'same'}), Digraph(graph_attr={'rank': 'same'})]
        nodes = [{}, {}] # type: List[Dict[Union[TLeft, TRight], str]]
        edges = [] # type: List [Tuple[str,str]]
        node_id = 0
        for (tail_part, tail), head_set in self.items():
            if tail not in nodes[tail_part]:
                name = 'node%d' % node_id
                nodes[tail_part][tail] = name
                subgraphs[tail_part].node(name, label=str(tail))
                node_id += 1
            for head_part, head in head_set:
                if head not in nodes[head_part]:
                    name = 'node%d' % node_id
                    nodes[head_part][head] = name
                    subgraphs[head_part].node(name, label=str(head))
                    node_id += 1
                edges.append((nodes[tail_part][tail], nodes[head_part][head]))
        graph.subgraph(subgraphs[0])
        graph.subgraph(subgraphs[1])
        for tail_node, head_node in edges:
            graph.edge(tail_node, head_node)
        return graph

    def find_cycle(self) -> List[Tuple[int,Union[TLeft,TRight]]]:
        visited = cast(Set[Tuple[int,Union[TLeft,TRight]]], set())
        for n in self:
            cycle = self._find_cycle(n, cast(List[Tuple[int,Union[TLeft,TRight]]], []), visited)
            if cycle:
                return cycle
        return cast(List[Tuple[int,Union[TLeft,TRight]]], [])

    def _find_cycle(self, node:Tuple[int,Union[TLeft,TRight]], path:List[Tuple[int,Union[TLeft,TRight]]], visited:Set[Tuple[int,Union[TLeft,TRight]]]) -> List[Tuple[int,Union[TLeft,TRight]]]:
        if node in visited:
            try:
                index = path.index(node)
                return path[index:]
            except ValueError:
                return cast(List[Tuple[int,Union[TLeft,TRight]]], [])

        visited.add(node)

        if node not in self:
            return cast(List[Tuple[int,Union[TLeft,TRight]]], [])

        for other in self[node]:
            cycle = self._find_cycle(other, path + [node], visited)
            if cycle:
                return cycle

        return cast(List[Tuple[int,Union[TLeft,TRight]]], [])

def enum_maximum_matchings_iter(graph: BipartiteGraph[TLeft, TRight, TEdge]) -> Iterator[Dict[TLeft, TRight]]:
    matching = graph.find_matching()
    if matching:
        yield matching
        yield from _enum_maximum_matchings_iter(graph, matching, _DirectedMatchGraph(graph, matching))

def _enum_maximum_matchings_iter(graph: BipartiteGraph[TLeft, TRight, TEdge], matching: Dict[TLeft, TRight], directed_match_graph: _DirectedMatchGraph[TLeft, TRight]) -> Iterator[Dict[TLeft, TRight]]:
    # Algorithm described in "Algorithms for Enumerating All Perfect, Maximum and Maximal Matchings in Bipartite Graphs"
    # By Takeaki Uno in "Algorithms and Computation: 8th International Symposium, ISAAC '97 Singapore, December 17-19, 1997 Proceedings"
    # See http://dx.doi.org/10.1007/3-540-63890-3_11

    # Step 1
    if len(graph) == 0:
        return

    # Step 2
    # Find a circle in the directed matching graph
    # Note that this circle alternates between nodes from the left and the right part of the graph
    raw_cycle = directed_match_graph.find_cycle()

    if raw_cycle:        
        # Make sure the circle "starts"" in the the left part
        # If not, start the circle from the second node, which is in the left part
        if raw_cycle[0][0] != LEFT:
            cycle = tuple([raw_cycle[-1][1]] + list(x[1] for x in raw_cycle[:-1]))
        else:
            cycle = tuple(x[1] for x in raw_cycle)

        # Step 3 - TODO: Properly find right edge
        edge = cast(Tuple[TLeft,TRight], cycle[:2])

        # Step 4
        # already done because we are not really finding the optimal edge

        # Step 5
        # Construct new matching M' by flipping edges along the cycle, i.e. change the direction of all the
        # edges in the circle
        new_match = matching.copy()
        for i in range(0, len(cycle), 2):
            new_match[cycle[i]] = cycle[i-1] # type: ignore

        yield new_match

        # Construct G+(e) and G-(e)
        graph_plus = graph.without_nodes(edge)
        graph_minus = graph.without_edge(edge)

        # Step 6
        # Recurse with the old matching M but without the edge e

        # Construct M\e
        match_minus_e = matching.copy()
        del match_minus_e[edge[0]]

        directed_match_graph_plus = _DirectedMatchGraph(graph_plus, match_minus_e)

        yield from _enum_maximum_matchings_iter(graph_plus, matching, directed_match_graph_plus)

        # Step 7
        # Recurse with the new matching M' but without the edge e
        directed_match_graph_minus = _DirectedMatchGraph(graph_minus, new_match)

        yield from _enum_maximum_matchings_iter(graph_minus, new_match, directed_match_graph_minus)

    else:
        # Step 8
        # Find feasible path of length 2 in D(graph, matching)
        # This path has the form left1 -> right -> left2
        # left1 must be in the left part of the graph and in matching
        # right must be in the right part of the graph
        # left2 is also in the left part of the graph and but must not be in matching
        left1 = None # type: TLeft
        left2 = None # type: TLeft
        right = None # type: TRight

        for part1, node1 in directed_match_graph:
            if part1 == LEFT and node1 in matching:
                left1 = cast(TLeft, node1)
                right = matching[left1]
                if (RIGHT, right) in directed_match_graph:
                    for _, node2 in directed_match_graph[(RIGHT, right)]:
                        if node2 not in matching:
                            left2 = cast(TLeft, node2)
                            break
                    if left2 is not None:
                        break

        if left2 is None:
            return

        # Construct M'
        # Exchange the direction of the path left1 -> right -> left2
        # to left1 <- right <- left2 in the new matching
        new_match = matching.copy()
        del new_match[left1]
        new_match[left2] = right

        yield new_match

        edge = (left2, right)

        # Construct M'\e
        new_match_minus_e = new_match.copy()
        del new_match_minus_e[left2]

        # Construct G+(e) and G-(e)
        graph_plus = graph.without_nodes(edge)
        graph_minus = graph.without_edge(edge)

        # Step 9
        yield from _enum_maximum_matchings_iter(graph_plus, new_match, _DirectedMatchGraph(graph_plus, new_match_minus_e))

        # Step 10
        yield from _enum_maximum_matchings_iter(graph_minus, matching, _DirectedMatchGraph(graph_minus, matching))


if __name__ == '__main__': # pragma: no cover
    def _main():
        from patternmatcher.expressions import (Arity, Operation, Symbol, Variable)

        f = Operation.new('f', Arity.variadic, 'f')
        g = Operation.new('g', Arity.variadic, 'g')

        a = Symbol('a')
        b = Symbol('b')
        x_ = Variable.dot('x')
        y_ = Variable.dot('y')

        G = BipartiteGraph()
        G[f(a),x_] = True
        G[f(b),x_] = True
        G[a,x_] = True
        G[b,x_] = True
        G[g(a),x_] = True
        G[f(a),y_] = True
        G[f(b),y_] = True
        G[a,y_] = True
        G[b,y_] = True
        G[g(a),y_] = True
        G[f(a),f(x_)] = True
        G[f(b),f(x_)] = True
        G[g(a),g(x_)] = True

        G.as_graph().render('G')

        m = G.find_matching()

        for k, v in m.items():
            print('%s: %s' % (k, v))

        DGM = _DirectedMatchGraph(G, m)
        DGM.as_graph().render('DGM')

        for m in _enum_maximum_matchings_iter(G, m, DGM):
            print('matching!')
            for kv in m.items():
                print('\t%s: %s' % kv)

    _main()
