# -*- coding: utf-8 -*-

from typing import Dict, List, Set, Tuple, TypeVar, Union

from graphviz import Digraph, Graph
from hopcroftkarp import HopcroftKarp

from patternmatcher.expressions import (Arity, Expression, Operation, Symbol,
                                        Variable, Wildcard)
from patternmatcher.functions import ReplacementRule
from patternmatcher.syntactic import DiscriminationNet

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

LEFT = 0
RIGHT = 1

TUTuple = TypeVar('TUTuple', bound=Tuple[T,U])
class BipartiteGraph(Dict[TUTuple,V]):
    def as_graph(self): # pragma: no cover
        graph = Graph()

        nodes1 = {}
        nodes2 = {}
        i = 0
        for (a, b), l in self.items():
            a = str(a)
            b = str(b)
            if a not in nodes1:
                name = 'a%d' % i
                nodes1[a] = name
                graph.node(name, label=a)
                i += 1
            if b not in nodes2:
                name = 'b%d' % i
                nodes2[b] = name
                graph.node(name, label=b)
                i += 1
            l = l is not True and str(l) or ''
            graph.edge(nodes1[a], nodes2[b], l)
        return graph

    def find_matching(self) -> Dict[T,Set[U]]:
        # The directed graph is represented as a dictionary of edges
        # The key is the tail of all edges which are represented by the value
        # The value is a set of heads for the all edges originating from the tail (key)
        # In addition, the graph stores which half of the bipartite graph a node originated from
        # to avoid problems when a value exists in both halfs.
        directed_graph = {} # type: Dict[Tuple(int, T),Set[Tuple(int,U)]]

        for (left, right) in self:
            tail = (LEFT, left)
            head = (RIGHT, right)
            if tail not in directed_graph:
                directed_graph[tail] = {head}
            else:
                directed_graph[tail].add(head)

        matching = HopcroftKarp(directed_graph).maximum_matching()

        # Filter out the partitions (LEFT and RIGHT) and only return the matching edges
        # that gor from LEFT to RIGHT
        return dict((tail[1], head[1]) for tail, head in matching.items() if tail[0] == LEFT)


class ManyToOneMatcher(object):
    def __init__(self, *patterns: Expression):
        self.patterns = patterns
        self.graphs = {}
        self.nets = {}

        for pattern in patterns:
            for operation, _ in pattern.preorder_iter(lambda e: isinstance(e, Operation) and e.commutative):
                if type(operation) not in self.graphs:
                    self.graphs[type(operation)] = set()
                expressions = [o for o in operation.operands if o.is_syntactic and o.head is not None]
                self.graphs[type(operation)].update(expressions)

        for g, exprs in self.graphs.items():
            net = DiscriminationNet()
            for expr in exprs:
                net.add(expr)
            self.nets[g] = net

def find_cycle(graph):
    #print('-'*100)
    #print('g', graph)
    visited = set()
    for n in graph:
        cycle = _find_cycle(graph, n, [], visited)
        if cycle:
            return cycle
    return []

def _find_cycle(graph: Dict[T, T], node:T, path:List[T], visited:Set[T]) -> bool:
    if node in visited:
        try:
            index = path.index(node)
            return path[index:]
        except ValueError:
            return []

    visited.add(node)

    if node not in graph:
        return []

    for other in graph[node]:
        cycle = _find_cycle(graph, other, path + [node], visited)
        if cycle:
            return cycle

    return []

def _graph_plus(G: BipartiteGraph[TUTuple, V], e: Tuple[T, U]) -> BipartiteGraph[TUTuple, V]:
    return BipartiteGraph([((n1, n2), v) for (n1, n2), v in G.items() if n1 != e[0] and n2 != e[1]])

def _graph_minus(G: BipartiteGraph[TUTuple, V], e: Tuple[T, U]) -> BipartiteGraph[TUTuple, V]:
    return BipartiteGraph([(e2, v) for e2, v in G.items() if e != e2])

DGMNode = TypeVar('DGMNode', bound=Tuple[int,Union[T,U]])
DGMEdge = TypeVar('DGMEdge', bound=Set[DGMNode])
class _DGM(Dict[DGMNode,DGMEdge]):
    def __init__(self, G: BipartiteGraph[TUTuple,V], M: Dict[T, U]) -> None:
        super(_DGM, self).__init__()
        for (n1, n2) in G:
            if n1 in M and M[n1] == n2:
                self[(LEFT, n1)] = {(RIGHT, n2)}
            else:
                if (RIGHT, n2) not in self:
                    self[(RIGHT, n2)] = set()
                self[(RIGHT, n2)].add((LEFT, n1))

def _make_graph(G): # pragma: no cover
    graph = Digraph()

    subgraphs = [Digraph(graph_attr={'rank': 'same'}), Digraph(graph_attr={'rank': 'same'})]
    nodes = [{}, {}]
    edges = []
    i = 0
    for (s, n), es in G.items():
        n = str(n)
        if n not in nodes[s]:
            name = 'node%d' % i
            nodes[s][n] = name
            subgraphs[s].node(name, label=n)
            i += 1
        for s2, e in es:
            e = str(e)
            if e not in nodes[s2]:
                name = 'node%d' % i
                nodes[s2][e] = name
                subgraphs[s2].node(name, label=e)
                i += 1
            edges.append((nodes[s][n], nodes[s2][e]))
    graph.subgraph(subgraphs[0])
    graph.subgraph(subgraphs[1])
    for a, b in edges:
        graph.edge(a, b)
    return graph

def enum_maximum_matchings_iter(G: Dict[T, Set[T]], M: Dict[T, T], DGM: Dict[T, Set[T]]):
    #print('-' * 100)
    #print ('G: ')
    #for x in G:
    #    print ('\t%s\t--\t%s' % x)
    #print ('M: ', ', '.join('%s: %s' % x for x in M.items()))
    #print ('D(G, M): ')
    #for x in DGM.items():
    #    print ('\t%s\t->\t%s' % x)
    #_make_graph(DGM).render('DGM%d' % enum_maximum_matchings_iter.DGM_COUNT)
    #enum_maximum_matchings_iter.DGM_COUNT += 1

    # Step 1
    if len(G) == 0:
        return

    # Step 2
    cycle = find_cycle(DGM)

    if cycle:
        if cycle[0][0] != LEFT:
            cycle = tuple([cycle[-1][1]] + list(x[1] for x in cycle[:-1]))
        else:
            cycle = tuple(x[1] for x in cycle)
        #print ('cycle: ', ' -> '.join(str(x) for x in cycle))

        # Step 3
        e = cycle[:2] # TODO: Properly find right edge
        #print('e: ', e)

        # Step 4
        # already done

        # Step 5
        # Construct new matching M' by changing edges along the cycle
        M2 = M.copy()
        for i in range(0, len(cycle), 2):
            M2[cycle[i]] = cycle[i-1]

        yield M2
        #print ('M2: ', ', '.join('%s: %s' % x for x in M2.items()))

        # Step 6
        Gp = _graph_plus(G, e)

        # Construct M\e
        M_minus_e = M.copy()
        del M_minus_e[e[0]]

        #print('GP: ')
        #for x in Gp:
        #    print ('\t%s\t--\t%s' % x)
        #print ('M_minus_e: ', ', '.join('%s: %s' % x for x in M_minus_e.items()))

        yield from enum_maximum_matchings_iter(Gp, M, _DGM(Gp, M_minus_e))

        # Step 7
        Gm = _graph_minus(G, e)

        yield from enum_maximum_matchings_iter(Gm, M2, _DGM(Gm, M2))

    else:
        # Step 8
        n1 = None
        n2 = None
        n = None

        # Find feasible path of length 2 in D(G, M)
        for k in DGM:
            if k[0] == LEFT and k[1] in M:
                for o in DGM[k]:
                    if o in DGM:
                        for _, o2 in DGM[o]:
                            if o2 not in M:
                                n1 = k[1]
                                n2 = o2
                                n = o[1]
                                break
                    if n1 is not None:
                        break
                if n1 is not None:
                    break

        if n1 is None:
            return

        #print ('path: %s -> %s -> %s' % (n1, n, n2))

        # Construct M'
        M2 = M.copy()
        del M2[n1]
        M2[n2] = n

        yield M2

        e = (n2, n)

        M3 = M2.copy()
        del M3[n2]

        # Construct G+(e) and G-(e)
        Gp = _graph_plus(G, e)
        Gm = _graph_minus(G, e)

        # Step 10
        yield from enum_maximum_matchings_iter(Gp, M2, _DGM(Gp, M3))

        # Step 10
        yield from enum_maximum_matchings_iter(Gm, M, _DGM(Gm, M))

enum_maximum_matchings_iter.DGM_COUNT = 0


if __name__ == '__main__': # pragma: no cover
    def _main3():
        # pylint: disable=invalid-name,bad-continuation
        Times = Operation.new('*', Arity.variadic, 'Times', associative=True, one_identity=True)
        Plus = Operation.new('+', Arity.variadic, 'Plus', associative=True, one_identity=True, commutative=True)
        Minus = Operation.new('-', Arity.unary, 'Minus')
        Inv = Operation.new('Inv', Arity.unary, 'Inv')
        Trans = Operation.new('T', Arity.unary, 'Trans')

        x_ = Variable.dot('x')
        x__ = Variable.plus('x')
        y_ = Variable.dot('y')
        y___ = Variable.star('y')
        z_ = Variable.dot('z')
        z___ = Variable.star('z')
        ___ = Wildcard.star()

        Zero = Symbol('0')
        Identity = Symbol('I')

        rules = [
            # --x -> x
            ReplacementRule(
                Minus(Minus(x_)),
                lambda x: x
            ),
            # -0 -> 0
            ReplacementRule(
                Minus(Zero),
                lambda: Zero
            ),
            # x + 0 -> x
            ReplacementRule(
                Plus(x_, Zero),
                lambda x: x
            ),
            # y + x - x -> y
            ReplacementRule(
                Plus(y_, x_, Minus(x_)),
                lambda x, y: y
            ),
            # x * 0 -> 0
            ReplacementRule(
                Times(___, Zero, ___),
                lambda: Zero
            ),
            # x * x^-1 -> I
            ReplacementRule(
                Times(y___, x_, Inv(x_), z___),
                lambda x, y, z: Times(*(y + [Identity] + z))
            ),
            # T(x) * T(x^-1) -> I
            ReplacementRule(
                Times(y___, Trans(x_), Trans(Inv(x_)), z___),
                lambda x, y, z: Times(*(y + [Identity] + z))
            ),
            # T(T(x)) -> x
            ReplacementRule(
                Trans(Trans(x_)),
                lambda x: x
            ),
            # T(0) -> 0
            ReplacementRule(
                Trans(Zero),
                lambda: Zero
            ),
            # T(I) -> I
            ReplacementRule(
                Trans(Identity),
                lambda: Identity
            ),
            # ((x)^-1)^-1 -> x
            ReplacementRule(
                Inv(Inv(x_)),
                lambda x: x
            ),
            # x * I -> x
            ReplacementRule(
                Times(y___, x_, Identity, z___),
                lambda x, y, z: Times(*(y + [x] + z))
            ),
            # I * x -> x
            ReplacementRule(
                Times(y___, Identity, x_, z___),
                lambda x, y, z: Times(*(y + [x] + z))
            ),
        ]

        matcher = ManyToOneMatcher(*(r.pattern for r in rules))

    def _main4():
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

        DGM = _DGM(G, m)

        #print(len(DGM))

        _make_graph(DGM).render('DGM')

        # print(find_cycle(DGM))

        for m in enum_maximum_matchings_iter(G, m, DGM):
            print('match!')
            for kv in m.items():
                print('\t%s: %s' % kv)
        #matches = list(enum_maximum_matchings_iter(G, m2, DGM))
        #print(matches)

    _main4()
