# -*- coding: utf-8 -*-
import itertools
import math
from graphviz import Digraph

from patternmatcher.expressions import Variable, Operation, Arity, Symbol, Wildcard, Atom

class OperationEnd(object):
    def __str__(self):
        return ')'

OPERATION_END = OperationEnd()

def flatterm_iter(expression):
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

class Flatterm(list):
    def __init__(self, expression):
        list.__init__(self, flatterm_iter(expression))

    def _term_str(self, term):
        if isinstance(term, type) and issubclass(term, Operation):
            return term.name + '('
        elif isinstance(term, Wildcard):
            return '*[%s,%s]' % (term.min_count, term.max_count)
        else:
            return str(term)

    def __str__(self):
        return ' '.join(map(self._term_str, self))

def get_flatterm(expression):
    if isinstance(expression, Variable):
        return get_flatterm(expression.expression)
    if isinstance(expression, Operation):
        name = expression.name + '('
        return list(itertools.chain([name], \
            itertools.chain.from_iterable(map(get_flatterm, expression.operands)), \
            [')']))
    if isinstance(expression, Symbol):
        return [expression.name]
    if isinstance(expression, Wildcard):
        # TODO: What about non-inf max with 0 min?
        wc = '*' * expression.min_count + (expression.max_count == math.inf and '+' or '')
        return [wc]

    raise NotImplementedError()

def is_wildcard(term):
    return term[0] == '*' and (term[-1] == '*' or term[-1] == '+')

def combine_wildcards(flattterm):
    flat = []

    for expr in flattterm:
        try:
            last = flat[-1]
            was_wildcard = is_wildcard(last)
            next_wildcard = is_wildcard(expr)

            if was_wildcard and next_wildcard:
                unbounded = last[-1] == '+' or expr[-1] == '+'
                min_count = (last[-1] != '+' and len(last) or len(last) - 1) + \
                            (expr[-1] != '+' and len(expr) or len(expr) - 1)
                new_wildcard = ('*' * min_count) + (unbounded and '+' or '')

                flat[-1] = new_wildcard
            else:
                flat.append(expr)
        except IndexError:
            flat.append(expr)

    return flat

def generate_net(pattern):
    last_node = None
    last_term = None
    last_wildcard = [None]
    last_symbol_after_wildcard = [None]
    fail_nodes = [None]
    all_same = [True]
    root = node = Node()
    root.id = 1
    flatterm = list(combine_wildcards(get_flatterm(pattern)))

    i = 2
    for j, term in enumerate(flatterm):
        last_node = node
        last_term = term
        if is_wildcard(term):
            for t in term:
                if t == '+':
                    node['*'] = node
                else:
                    node['*'] = Node()
                    node['*'].id = i
                    i += 1
                    node = node['*']
            if term[-1] == "+":
                last_wildcard[-1] = node
                last_symbol_after_wildcard[-1] = None
                all_same[-1] = True
        else:
            node[term] = Node()
            node[term].id = i
            i += 1
            node = node[term]
            if last_symbol_after_wildcard[-1] is None:
                last_symbol_after_wildcard[-1] = term
            if term != last_symbol_after_wildcard[-1]:
                all_same[-1] = False
            if term[-1] == '(':
                last_wildcard.append(None)
                last_symbol_after_wildcard.append(None)
                fail_nodes.append(None)
                all_same.append(True)
            if term == ')':
                last_wildcard.pop()
                last_symbol_after_wildcard.pop()
                fail_nodes.pop()
                all_same.pop()
            if last_wildcard[-1] is not None:
                try:
                    next_term = flatterm[j+1]
                except IndexError:
                    next_term = None
                if next_term is not None and not is_wildcard(next_term):
                    node['*'] = last_wildcard[-1]
                    if next_term != last_symbol_after_wildcard[-1]:
                        if all_same[-1] and next_term == ')':
                            node[last_symbol_after_wildcard[-1]] = node
                        else:
                            node[last_symbol_after_wildcard[-1]] = last_wildcard[-1][last_symbol_after_wildcard[-1]]
            elif any(x is not None for x in last_wildcard):
                if fail_nodes[-1] is None:
                    fail_nodes[-1] = fn = Node()
                    fn['*'] = fn
                    for w in reversed(last_wildcard[:-1]):
                        fn[')'] = w or Node()
                        if w is None:
                            fn = fn[')']
                            fn['*'] = fn
                        else:
                            break
                node['*'] = fail_nodes[-1]

    last_node[last_term] = [pattern]

    return root

def product_net(node1, node2):
    root = Node()
    nodes = {(node1.id, node2.id): root}
    queue = [(node1, node2)]
    
    while len(queue) > 0:
        n1, n2 = queue.pop(0)
        keys = set()
        if n1 is not None:
            keys.update(n1.keys())
            id1 = n1.id
        else:
            id1 = 0
        if n2 is not None:
            keys.update(n2.keys())
            id2 = n2.id
        else:
            id2 = 0

        keys = list(keys)
        keys.sort()

        node = nodes[(id1, id2)]

        for k in list(keys):
            if n1 is not None:
                try:
                    try:
                        t1 = n1[k]
                    except (KeyError):
                        t1 = n1['*']
                        if k == ')':
                            raise KeyError()
                    id1 = t1.id
                except KeyError:
                    t1 = None
                    id1 = 0
                except AttributeError:
                    id1 = 0
            else:
                t1 = None
                id1 = 0
            if n2 is not None:
                try:
                    try:
                        t2 = n2[k]
                    except (KeyError):
                        if k == ')':
                            raise KeyError()
                        t2 = n2['*']
                    id2 = t2.id
                except KeyError:
                    t2 = None
                    id2 = 0
                except AttributeError:
                    id2 = 0
            else:
                t2 = None
                id2 = 0
            
            if id1 != 0 or id2 != 0:
                if (id1, id2) not in nodes:
                    nt = Node()
                    nt.id = 100 * id1 + id2
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
                    dot.edge('n%s' % node.id, 'n%s' % other.id, label)
                    if other.id not in nodes:
                        queue.append(other)
                else:
                    dot.edge('n%s' % node.id, 'l%s' % id(other), label)

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

    ft = Flatterm(f(z, g(a)))
    print(ft)

    #print (' '.join(str(x) for x in combine_wildcards(get_flatterm(expr))))
    net = DiscriminationNet()

    net1 = generate_net(expr1)
    net2 = generate_net(expr2)
    net3 = generate_net(expr3)
    net4 = generate_net(expr4)

    net5 = product_net(net2, net3)
    net._net = net5 # product_net2(net4, net3)

    #net.add(expr1)
    #net.add(expr2)

    graph = net.dot()
    #print(graph.source)

    graph.render()