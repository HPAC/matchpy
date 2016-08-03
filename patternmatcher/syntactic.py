# -*- coding: utf-8 -*-
import math

from graphviz import Digraph

from patternmatcher.expressions import (Arity, Atom, Operation, Symbol,
                                        Variable, Wildcard)


class _OperationEnd(object):
    def __str__(self):
        return ')'

OPERATION_END = _OperationEnd()

def is_operation(term):
    return isinstance(term, type) and issubclass(term, Operation)

class Flatterm(list):
    def __init__(self, expression):
        list.__init__(self, self._combined_wildcards_iter(self._flatterm_iter(expression)))

    def _flatterm_iter(self, expression):
        """Generator that yields the atoms of the expressions in prefix notation with operation end markers.
        
        See :class:`Flatterm` for details of the flatterm form.
        """
        if isinstance(expression, Variable):
            yield from self._flatterm_iter(expression.expression)
        elif isinstance(expression, Operation):
            yield type(expression)
            for operand in expression.operands:
                yield from self._flatterm_iter(operand)
            yield OPERATION_END
        elif isinstance(expression, Atom):
            yield expression
        else:
            raise TypeError()

    def _combined_wildcards_iter(self, flatterm):
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
        if last_wildcard is not None:
            yield last_wildcard

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
        self.last_wildcard = None # last unbounded wildcard at the current level of operation nesting (if any)
        self.symbol_after = None # symbol after the last wildcard
        self.all_same = True # True iff all symbols until now have been the same after the last unbounded wildcard
        self.fail_node = None # The failure node for the current level of operation nesting


class _NodeQueueItem(object):
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        try:
            self.id1 = node1.id
        except AttributeError:
            self.id1 = 0
        try:
            self.id2 = node2.id
        except AttributeError:
            self.id2 = 0
        self.depth = 0
        self.fixed = 0

    @property
    def keys(self):
        keys = set()
        if self.node1 is not None and self.fixed != 1:
            keys.update(self.node1.keys())
        if self.node2 is not None and self.fixed != 2:
            keys.update(self.node2.keys())
        if self.fixed != 0:
            if self.fixed == 1 and self.node2 is None:
                keys.add(OPERATION_END)
            elif self.fixed == 2 and self.node1 is None:
                keys.add(OPERATION_END)
            keys.add(Wildcard)
        return keys

class _Node(dict):
    _id = 1

    def __init__(self):
        super().__init__(self)
        self.id = _Node._id
        _Node._id += 1

    def _term_str(self, term):
        if is_operation(term):
            return term.name + '('
        elif term == Wildcard:
            return '*'
        else:
            return str(term)

    def _val_str(self, value):
        if value is self:
            return 'self'
        elif isinstance(value, list):
            return repr(list(map(str, value)))
        else:
            return str(value)

    def __repr__(self):
        return '{NODE %s}' % (', '.join('%s:%s' % (self._term_str(k), self._val_str(v)) for k, v in self.items()))

class DiscriminationNet(object):
    def __init__(self):
        self._net = _Node()

    def add(self, pattern):
        net = DiscriminationNet._generate_net(pattern)
        self._net = DiscriminationNet._product_net(self._net, net)
    
    @staticmethod
    def _generate_net(pattern):
        """Generates a DFA matching the given pattern."""
        last_node = None
        last_term = None
        # Capture the last unbounded wildcard for every level of operation nesting on a stack
        # Used to add backtracking edges in case the "match" fails later 
        wildcard_states = [_WildcardState()]
        root = node = _Node()
        flatterm = Flatterm(pattern)

        #i = 2
        for j, term in enumerate(flatterm):
            last_node = node
            last_term = term
            state = wildcard_states[-1]
            # For wildcards, generate a chain of #min_count Wildcard edges
            # If the wildcard is unbounded (max_count = math.inf),
            # add a wildcard self loop at the end
            if isinstance(term, Wildcard):            
                last_term = Wildcard
                for _ in range(term.min_count):
                    node[Wildcard] = _Node()
                    node = node[Wildcard]
                if term.max_count == math.inf:
                    node[Wildcard] = node
                    # set wildcard state to this reference this wildcard
                    state.last_wildcard = node
                    state.symbol_after = None
                    state.all_same = True
            else:
                node[term] = _Node()
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
                            state.fail_node = fn = _Node()
                            fn[Wildcard] = fn
                            for last_state in reversed(wildcard_states[:-1]):
                                fn[OPERATION_END] = last_state.last_wildcard or last_state.fail_node or _Node()
                                if last_state.fail_node is not None or last_state.last_wildcard is not None:
                                    break
                                last_state.fail_node = fn = fn[OPERATION_END]
                                fn[Wildcard] = fn
                        node[Wildcard] = state.fail_node

        last_node[last_term] = [pattern]

        return root

    @staticmethod
    def _product_net(node1, node2):
        def get_child(node, key, fixed):
            if fixed:
                return node, False
            if node is not None:
                try:
                    try:
                        return node[key], False
                    except (KeyError):
                        if key == OPERATION_END:
                            return None, False
                        return node[Wildcard], True
                except KeyError:
                    return None, False
            return None, False

        root = _Node()
        nodes = {(node1.id, node2.id): root}
        queue = [_NodeQueueItem(node1, node2)]
        
        while len(queue) > 0:
            state = queue.pop(0)
            node = nodes[(state.id1, state.id2)]

            for k in list(state.keys):
                t1, with_wildcard1 = get_child(state.node1, k, state.fixed == 1)
                t2, with_wildcard2 = get_child(state.node2, k, state.fixed == 2)

                child_state = _NodeQueueItem(t1, t2)
                child_state.depth = state.depth
                child_state.fixed = state.fixed

                if is_operation(k):
                    if state.fixed:
                        child_state.depth += 1
                    elif with_wildcard1:
                        child_state.fixed = 1
                        child_state.depth = 1
                        child_state.node1 = state.node1
                    elif with_wildcard2:
                        child_state.fixed = 2
                        child_state.depth = 1
                        child_state.node2 = state.node2
                elif k == OPERATION_END and state.fixed:
                    child_state.depth -= 1

                    if child_state.depth == 0:
                        if child_state.fixed == 1:
                            child_state.node1 = child_state.node1[Wildcard]
                            try:
                                child_state.id1 = child_state.node1.id
                            except AttributeError:
                                child_state.id1 = 0
                        elif child_state.fixed == 2:
                            child_state.node2 = child_state.node2[Wildcard]
                            try:
                                child_state.id2 = child_state.node2.id
                            except AttributeError:
                                child_state.id2 = 0
                        else:
                            assert False # unreachable
                        child_state.fixed = 0
                
                if child_state.id1 != 0 or child_state.id2 != 0:
                    if (child_state.id1, child_state.id2) not in nodes:
                        nodes[(child_state.id1, child_state.id2)] = _Node()
                        queue.append(child_state)
                    
                    node[k] = nodes[(child_state.id1, child_state.id2)]
                else:
                    if type(child_state.node1) == list and type(child_state.node2) == list:
                        node[k] = child_state.node1 + child_state.node2               
                    elif type(child_state.node1) == list:
                        node[k] = child_state.node1              
                    elif type(child_state.node2) == list:
                        node[k] = child_state.node2

        return root

    def match(self, expression):
        node = self._net
        depth = 0
        for term in Flatterm(expression):
            if depth > 0:
                if is_operation(term):
                    depth += 1
                elif term == OPERATION_END:
                    depth -= 1
            else:
                try:
                    try:
                        node = node[term]
                    except KeyError:
                        if is_operation(term):
                            depth = 1
                        node = node[Wildcard]
                except KeyError:
                    return []

                if type(node) == list:
                    return node

        assert type(node) == list
        return node

    def _term_str(self, term):
        if is_operation(term):
            return term.name + '('
        elif term == Wildcard:
            return '*'
        else:
            return str(term)

    def as_graph(self):
        dot = Digraph()

        nodes = set()
        queue = [self._net]
        while queue:
            node = queue.pop(0)
            nodes.add(node.id)
            dot.node('n%s' % node.id, '', {'shape': 'point'})

            for next_node in node.values():
                if isinstance(next_node, _Node):
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
                if isinstance(other, _Node):
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
    net.add(expr1)
    net.add(expr2)
    net.add(expr3)
    net.add(expr4)

    graph = net.as_graph()

    graph.render()