# -*- coding: utf-8 -*-
import itertools
import math

from graphviz import Digraph

from patternmatcher.expressions import (Arity, Atom, Operation, Symbol,
                                        Variable, Wildcard)


class OperationEnd(object):
    def __str__(self):
        return ')'

OPERATION_END = OperationEnd()

def is_operation(term):
    return isinstance(term, type) and issubclass(term, Operation)

def flatterm_iter(expression):
    """Generator that yields the atoms of the expressions in prefix notation with operation end markers.
    
    See :class:`Flatterm` for details of the flatterm form.
    """
    if isinstance(expression, Variable):
        yield from flatterm_iter(expression.expression)
    elif isinstance(expression, Operation):
        yield type(expression)
        for operand in expression.operands:
            yield from flatterm_iter(operand)
        yield OPERATION_END
    elif isinstance(expression, Atom):
        yield expression
    else:
        raise TypeError()

def combined_wildcards_iter(flatterm):
    """Combines consecutive wildcards in a flatterm into a single one"""
    last_wildcard = None
    for term in flatterm:
        if isinstance(term, Wildcard):
            if last_wildcard is not None:
                last_wildcard = Wildcard(last_wildcard.min_count + term.min_count, last_wildcard.max_count + term.max_count)
            else:
                last_wildcard = term
        else:
            if last_wildcard is not None:
                yield last_wildcard
                last_wildcard = None
            yield term

class Flatterm(list):
    def __init__(self, expression):
        list.__init__(self, flatterm_iter(expression))

    def _term_str(self, term):
        if is_operation(term):
            return term.name + '('
        elif isinstance(term, Wildcard):
            return '*[%s,%s]' % (term.min_count, term.max_count)
        else:
            return str(term)

    def __str__(self):
        return ' '.join(map(self._term_str, self))

class _WildcardState:
    def __init__(self):
        self.last_wildcard = None
        self.symbol_after = None
        self.all_same = True
        self.fail_node = None

def generate_net(pattern):
    """"""
    last_node = None
    last_term = None
    # Capture the last unbounded wildcard for every level of operation nesting on a stack
    # Used to add backtracking edges in case the "match" fails later 
    wildcard_states = [_WildcardState()]
    root = node = Node()
    flatterm = list(combined_wildcards_iter(flatterm_iter(pattern)))

    #i = 2
    for j, term in enumerate(flatterm):
        last_node = node
        last_term = term
        state = wildcard_states[-1]
        # For wildcards, generate a chain of #min_count Wildcard edges
        # If the wildcard is unbounded (max_count = math.inf),
        # add a wildcard self loop at the end
        if isinstance(term, Wildcard):
            for _ in range(term.min_count):
                node[Wildcard] = Node()
                node = node[Wildcard]
            if term.max_count == math.inf:
                node[Wildcard] = node
                # set wildcard state to this reference this wildcard
                state.last_wildcard = node
                state.symbol_after = None
                state.all_same = True
        else:
            node[term] = Node()
            node = node[term]
            if state.symbol_after is None:
                state.symbol_after = term
            if term != state.symbol_after:
                state.all_same = False
            if is_operation(term):
                wildcard_states.append(_WildcardState())
            if term == OPERATION_END:
                wildcard_states.pop()

            state = wildcard_states[-1]
            try:
                next_term = flatterm[j+1]
            except IndexError:
                next_term = None

            # Potentially, backtracking wildcard edges have to be added
            if next_term is not None and not isinstance(next_term, Wildcard):
                # If there was an unbounded wildcard inside the current operation,
                # add a backtracking wildcard edge to it
                if state.last_wildcard is not None:
                    node[Wildcard] = state.last_wildcard
                    # Also add an edge for the symbol directly after the wildcard to
                    # its respecitive node (or as a self loop if all symbols are the same)
                    if next_term != state.symbol_after:
                        if state.all_same and next_term == OPERATION_END:
                            node[state.symbol_after] = node
                        else:
                            node[state.symbol_after] = state.last_wildcard[state.symbol_after]
                # If there was an unbounded wildcard inside a parent operation of the current one,
                # an additional fail state is needed, that eventually backtracks to the wildcard
                # Every level of operation nesting gets its own fail node until the level of the
                # wildcard is reached
                elif any(s.last_wildcard is not None for s in wildcard_states):
                    if state.fail_node is None:
                        state.fail_node = fn = Node()
                        fn[Wildcard] = fn
                        for last_state in reversed(wildcard_states[:-1]):
                            fn[OPERATION_END] = last_state.last_wildcard or last_state.fail_node or Node()
                            if last_state.fail_node is not None or last_state.last_wildcard is not None:
                                break
                            last_state.fail_node = fn = fn[OPERATION_END]
                            fn[Wildcard] = fn
                    node[Wildcard] = state.fail_node

    last_node[last_term] = [pattern]

    return root

class _NodeQueueItem(object):
    def __init__(self, node1, node2, id1, id2):
        self.node1 = node1
        self.node2 = node2
        self.id1 = id1
        self.id2 = id2
        self.depth = 0
        self.fixed = 0

    @property
    def keys(self):
        keys = set()
        if self.node1 is not None:
            keys.update(self.node1.keys())
        if self.node2 is not None:
            keys.update(self.node2.keys())
        return keys

def product_net(node1, node2):
    def get_keys_and_ids(*nodes):
        keys = set()
        ids = []
        for node in nodes:
            if node is not None:
                keys.update(node.keys())
                ids.append(node.id)
            else:
                ids.append(0)

        return keys, tuple(ids)

    def get_child_with_id(node, key):
        if node is not None:
            try:
                try:
                    child = node[key]
                except (KeyError):
                    if key == OPERATION_END:
                        return None, 0
                    child = node[Wildcard]
                return child, child.id
            except KeyError:
                return None, 0
            except AttributeError:
                return child, 0
        return None, 0

    root = Node()
    nodes = {(node1.id, node2.id): root}
    queue = [(node1, node2)]
    
    while len(queue) > 0:
        n1, n2 = queue.pop(0)
        keys, (id1, id2) = get_keys_and_ids(n1, n2)

        node = nodes[(id1, id2)]

        for k in list(keys):

            t1, id1 = get_child_with_id(n1, k)
            t2, id2 = get_child_with_id(n2, k)
            
            if id1 != 0 or id2 != 0:
                if (id1, id2) not in nodes:
                    nt = Node()
                    nodes[(id1, id2)] = nt
                    queue.append((t1, t2))
                
                node[k] = nodes[(id1, id2)]
            else:
                if type(t1) == list and type(t2) == list:
                    node[k] = t1 + t2               
                elif type(t1) == list:
                    node[k] = t1              
                elif type(t2) == list:
                    node[k] = t2

    return root

class Node(dict):
    _id = 1

    def __init__(self):
        super().__init__(self)
        self.id = Node._id
        Node._id += 1

class DiscriminationNet(object):
    def __init__(self):
        self._net = Node()

    def add(self, pattern):
        pass

    def match(self, expression):
        pass

    def _term_str(self, term):
        if is_operation(term):
            return term.name + '('
        elif term == Wildcard:
            return '*'
        else:
            return str(term)

    def dot(self):
        dot = Digraph()

        nodes = set()
        queue = [self._net]
        while queue:
            node = queue.pop(0)
            nodes.add(node.id)
            dot.node('n%s' % node.id, '', {'shape': 'point'})

            for next_node in node.values():
                if isinstance(next_node, Node):
                    if next_node.id not in nodes:
                        queue.append(next_node)
                else:
                    l = '\n'.join(str(x) for x in next_node)
                    dot.node('l%s' % id(next_node), l, {'shape': 'plaintext'})

        nodes = set()
        queue = [self._net]
        while queue:
            node = queue.pop(0)
            if node.id in nodes:
                continue
            nodes.add(node.id)

            for (label, other) in node.items():
                if isinstance(other, Node):
                    dot.edge('n%s' % node.id, 'n%s' % other.id, self._term_str(label))
                    if other.id not in nodes:
                        queue.append(other)
                else:
                    dot.edge('n%s' % node.id, 'l%s' % id(other), self._term_str(label))

        return dot

if __name__ == '__main__':
    f = Operation.new('f', arity=Arity.binary)
    g = Operation.new('g', arity=Arity.unary)
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    x = Variable.dot('x')
    y = Variable.star('y')
    z = Variable.plus('z')

    #expr1 = f(a, g(b))
    expr1 = f(a)
    #expr2 = f(x, z)
    expr2 = f(z)
    #expr3 = f(z, g(a))
    expr3 = f(g(a))
    expr4 = f(a, z)

    # EDGE CASE: f(z, a, x, b)

    net = DiscriminationNet()

    net1 = generate_net(expr1)
    net2 = generate_net(expr2)
    net3 = generate_net(expr3)
    net4 = generate_net(expr4)

    net5 = product_net(net1, net4)
    net._net = net5 # product_net2(net4, net3)

    graph = net.dot()
    #print(graph.source)

    graph.render()