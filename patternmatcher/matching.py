# -*- coding: utf-8 -*-

from patternmatcher.expressions import Expression, Operation, Variable, Symbol, Arity, Wildcard
from patternmatcher.functions import ReplacementRule
from patternmatcher.syntactic import DiscriminationNet

from typing import Dict, TypeVar, Set, List, Tuple, Generic
from graphviz import Graph, Digraph

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

class UndirectedGraph(dict):
    def __getitem__(self, key):
        return dict.__getitem__(self, tuple(sorted(key)))

    def __setitem__(self, key, value):
        dict.__setitem__(self, tuple(sorted(key)), value)

    def __delitem__(self, key):
        dict.__delitem__(self, tuple(sorted(key)))

    def as_graph(self):
        graph = Graph()

        nodes = {}
        i = 0
        for (a, b), l in self.items():
            a = str(a)
            b = str(b)
            if a not in nodes:
                name = 'node%d' % i
                nodes[a] = name
                graph.node(name, label=a)
                i += 1
            if b not in nodes:
                name = 'node%d' % i
                nodes[b] = name
                graph.node(name, label=b)
                i += 1
            graph.edge(nodes[a], nodes[b], str(l))
        return graph

TUTuple = TypeVar('TUTuple', bound=Tuple[T,U])
class BipartiteGraph(Dict[TUTuple,V]):
    def as_graph(self):
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

def find_circle(graph):
    #print('-'*100)
    #print('g', graph)
    visited = set()
    for n in graph:
        circle = _find_circle(graph, n, [], visited)
        if circle:
            return circle
    return []

def _find_circle(graph: Dict[T, T], node:T, path:List[T], visited:Set[T]) -> bool:
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
        circle = _find_circle(graph, other, path + [node], visited)
        if circle:
            return circle

    return []

def _graph_plus(G: UndirectedGraph, e: Tuple[T, T]):
    return UndirectedGraph([((n1, n2), v) for (n1, n2), v in G.items() if n1 not in e and n2 not in e])

def _graph_minus(G: UndirectedGraph, e: Tuple[T, T]):
    return UndirectedGraph([(e2, v) for e2, v in G.items() if e != e2])

def _DGM(G: BipartiteGraph[TUTuple,V], M: Dict[T, U]):
    DGM = {}
    for (n1, n2) in G:
        if n1 in M and M[n1] == n2:
            if (1, n2) not in DGM:
                DGM[(1, n2)] = set()
            DGM[(1, n2)].add((0, n1))
        else:
            if (0, n1) not in DGM:
                DGM[(0, n1)] = set()
            DGM[(0, n1)].add((1, n2))
    return DGM

def _make_graph(G):
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
    _make_graph(DGM).render('DGM%d' % enum_maximum_matchings_iter.DGM_COUNT)
    enum_maximum_matchings_iter.DGM_COUNT += 1

    # Step 1
    if len(G) == 0:
        return
    
    # Step 2
    circle = find_circle(DGM)

    if circle:
        if circle[0][0] != 0:
            circle = tuple([circle[-1][1]] + list(x[1] for x in circle[:-1]))
        else:
            circle = tuple(x[1] for x in circle)
        #print ('circle: ', ' -> '.join(str(x) for x in circle))

        # Step 3
        e = circle[:2] # TODO: Properly find right edge
        #print('e: ', e)
        
        # Step 4
        # already done
        
        # Step 5
        M2 = M.copy()
        for i in range(0, len(circle), 2):
            M2[circle[i]] = circle[i-1]

        yield M2
        #print ('M2: ', ', '.join('%s: %s' % x for x in M2.items()))
        
        # Step 6
        Gp = _graph_plus(G, e)
        M3 = M.copy()
        del M3[e[0]]

        #print('GP: ')
        #for x in Gp:
        #    print ('\t%s\t--\t%s' % x)
        #print ('M3: ', ', '.join('%s: %s' % x for x in M3.items()))

        yield from enum_maximum_matchings_iter(Gp, M, _DGM(Gp, M3))

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
            if k[0] == 0 and k[1] not in M:
                for o in DGM[k]:
                    if o in DGM:
                        for _, o2 in DGM[o]:
                            n1 = k[1]
                            n2 = o2
                            n = o[1]
                            break
                    if n1 is not None:
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
        del M2[n2]
        M2[n1] = n

        yield M2

        e = (n1, n)

        # Construct G+(e) and G-(e)
        Gp = _graph_plus(G, e)
        Gm = _graph_minus(G, e)

        # Step 10
        yield from enum_maximum_matchings_iter(Gp, M2, {})

        # Step 10
        yield from enum_maximum_matchings_iter(Gm, M, {})

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
                lambda x, y, z: Times(*y, Identity, *z)
            ),
            # T(x) * T(x^-1) -> I
            ReplacementRule(
                Times(y___, Trans(x_), Trans(Inv(x_)), z___),
                lambda x, y, z: Times(*y, Identity, *z)
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
                lambda x, y, z: Times(*y, x, *z)
            ),
            # I * x -> x
            ReplacementRule(
                Times(y___, Identity, x_, z___),
                lambda x, y, z: Times(*y, x, *z)
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

        from hopcroftkarp import HopcroftKarp

        G2 = {}

        for (n1, n2) in G:
            if (0, n1) not in G2:
                G2[(0, n1)] = set([(1, n2)])
            else:
                G2[(0, n1)].add((1, n2))

        h = HopcroftKarp(G2)
        m = h.maximum_matching()
        
        m2 = dict([(k[1], v[1]) for k, v in m.items() if k[0] == 0])
        
        for k, v in m2.items():
            print('%s: %s' % (k, v))

        DGM = _DGM(G, m2)

        #print(len(DGM))

        _make_graph(DGM).render('DGM')

        # print(find_circle(DGM))

        for m in enum_maximum_matchings_iter(G, m2, DGM):
            print('match!')
            for kv in m.items():
                print('\t%s: %s' % kv)
        #matches = list(enum_maximum_matchings_iter(G, m2, DGM))
        #print(matches)

    _main4()

    #_main3()

    #print(_graph_minus())
