# -*- coding: utf-8 -*-

from collections import namedtuple, Counter
from typing import Dict, Set, cast, Tuple, Union, List

from patternmatcher.bipartite import (BipartiteGraph,
                                      enum_maximum_matchings_iter)
from patternmatcher.expressions import (Arity, Expression, Operation, Symbol,
                                        Variable, List)
from patternmatcher.syntactic import DiscriminationNet
from patternmatcher.utils import fixed_integer_vector_iter, minimum_integer_vector_iter

CommutativeParts = namedtuple('CommutativeParts', ['constant', 'syntactic', 'variable', 'fixed'])

class Substitution(Dict[str, Union[List[Expression], Expression]]):
    @staticmethod
    def _match_value_repr_str(value: Union[List[Expression], Expression]) -> str: # pragma: no cover
        if isinstance(value, list):
            return '(%s)' % (', '.join(str(x) for x in value))
        return str(value)

    def __str__(self): # pragma: no cover
        return ', '.join('%s ← %s' % (k, self._match_value_repr_str(v)) for k, v in self.items())


class ManyToOneMatcher(object):
    def __init__(self, *patterns: Expression) -> None:
        self.patterns = patterns
        self.graphs = {} # type: Dict[type, Set[Expression]]
        self.nets = {} # type: Dict[type, DiscriminationNet]
        self.bipartites  = {} # type: Dict[type, BipartiteGraph]

        for pattern in patterns:
            self._extract_subexpressions(pattern, False, self.graphs)

        for g, exprs in self.graphs.items():
            net = DiscriminationNet()
            for expr in exprs:
                net.add(expr)
            self.nets[g] = net

    def match(self, expression):
        subexpressions = self._extract_subexpressions(expression, True)
        print(subexpressions)
        bipartites = {}
        for t, es in subexpressions.items():
            if t in self.nets:
                bipartites[t] = BipartiteGraph()
                for e in es:
                    for p in self.nets[t].match(e):
                        bipartites[t][e,p] = True
                bipartites[t].as_graph().render(t.__name__)

    @staticmethod
    def _extract_subexpressions(expression: Expression, include_constant: bool, subexpressions:Dict[type, Set[Expression]]=None) -> Dict[type, Set[Expression]]:
        if subexpressions is None:
            subexpressions = {}
        for subexpr, _ in expression.preorder_iter(lambda e: isinstance(e, Operation) and e.commutative):
            operation = cast(Operation, subexpr)
            if type(operation) not in subexpressions:
                subexpressions[type(operation)] = set()
            parts = ManyToOneMatcher._split_parts(operation.operands)
            expressions = parts.syntactic
            if include_constant:
                expressions += parts.constant
            subexpressions[type(operation)].update(expressions)
        return subexpressions

    @staticmethod
    def _split_parts(expressions: List[Expression]) -> CommutativeParts:
        constants  = []
        syntactics = []
        variables  = []
        fixed      = []

        for expression in sorted(expressions):
            if expression.is_constant:
                constants.append(expression)
            elif expression.head is None:
                wc = expression
                while isinstance(wc, Variable):
                    wc = wc.expression
                if wc.fixed_size:
                    fixed.append((expression, wc.min_count))
                else:
                    variables.append((expression, wc.min_count))
            elif expression.is_syntactic:
                syntactics.append(expression)
            else:
                fixed.append((expression, 1))

        return CommutativeParts(constants, syntactics, variables, fixed)

class CommutativeMatcher(object):
    def __init__(self) -> None:
        self.patterns = set() # type: Set[Expression]
        self.expressions = set() # type: Set[Expression]
        self.net = DiscriminationNet()
        self.bipartite = BipartiteGraph()

    def add_pattern(self, pattern: Expression):
        if pattern not in self.patterns:
            self.patterns.add(pattern)
            self.net.add(pattern)

    def add_expression(self, expression: Expression):
        if expression not in self.expressions:
            self.expressions.add(expression)
            for pattern in self.net.match(expression):
                subst = Substitution()
                if self._extract_substitution(expression, pattern, subst):
                    self.bipartite[expression, pattern] = subst

    def match(self, expression: List[Expression], pattern: CommutativeParts, label):
        if len(pattern.constant) > len(expression):
            return

        constants = expression[:]
        for const_pattern in pattern.constant:
            try:
                constants.remove(const_pattern)
            except ValueError:
                return

        constants, syntactics = self.split_expressions(constants)

        #print(len(syntactics), len(pattern.syntactic), len(pattern.fixed))

        if len(pattern.syntactic) > len(syntactics):
            return

        #if not pattern.variable and len(pattern.syntactic) != len(syntactics):
        #    return

        if pattern.syntactic:
            subgraph = self.bipartite.limited_to(syntactics, pattern.syntactic)
            subgraph.as_graph().render(label + '.gv')
            match_iter = enum_maximum_matchings_iter(subgraph)
            try:
                matching = next(match_iter)
            except StopIteration:
                return
            if len(matching) < len(pattern.syntactic):
                return
                
            yield from self._matches_from_matching(matching, constants, syntactics, pattern)
            for matching in match_iter:
                yield from self._matches_from_matching(matching, constants, syntactics, pattern)

    def _matches_from_matching(self, matching, constants, syntactics, pattern):
        substs = [self.bipartite[s] for s in matching.items()]
        subst = self._unify_substitutions(*substs)
        if subst is None:
            return
        remaining = constants + [s for s in syntactics if s not in matching]
        fixed_count = sum(c for _, c in pattern.fixed)
        var_min_count = sum(c for _, c in pattern.variable)
        if len(remaining) < fixed_count + var_min_count:
            return
        real_fixed = []
        fixed_vars = []
        for f, count in pattern.fixed:
            if isinstance(f, Variable):
                if f.name in subst:
                    try:
                        remaining.remove(subst[f.name])
                    except ValueError:
                        return
                else:
                    fixed_vars.append((f, count))
            else:
                real_fixed.append(f)

        factories = [self._fixed_expr_factory(e) for e in real_fixed] + \
                    [self._fixed_var_iter_factory(v.name, l) for v, l in fixed_vars] + \
                    [self._sequence_var_iter_factory(v.name, l) for v, l in pattern.variable]

        expr_counter = Counter(remaining)

        for rem_expr, subst in iterator_chain((expr_counter, subst), *factories):
            if not rem_expr:
                yield subst

    @staticmethod
    def _fixed_expr_factory(expression):
        def factory(expressions, substitution):
            for expr in expressions:
                if expr.head == expression.head:
                    yield expressions - Counter({expr: 1}), substitution

        return factory

    @staticmethod
    def _fixed_var_iter_factory(variable, length):
        def factory(expressions, substitution):
            if variable in substitution:
                existing = Counter(substitution[variable])
                if existing - expressions:
                    return
                yield expressions - existing, substitution
            else:
                if length == 1:
                    for expr in expressions:
                        new_substitution = Substitution(substitution)
                        new_substitution[variable] = expr
                        yield expressions - Counter({expr: 1}), new_substitution
                else:
                    exprs_with_counts = list(expressions.items())
                    counts = tuple(c for _, c in exprs_with_counts)
                    for subset in fixed_integer_vector_iter(counts, length):
                        sub_counter = Counter(dict((exprs_with_counts[i][0], c) for i, c in enumerate(subset)))
                        new_substitution = Substitution(substitution)
                        new_substitution[variable] = list(sorted(sub_counter.elements()))
                        yield expressions - sub_counter, new_substitution

        return factory

    @staticmethod
    def _sequence_var_iter_factory(variable, minimum):
        def factory(expressions, substitution):
            if variable in substitution:
                existing = Counter(substitution[variable])
                if existing - expressions:
                    return
                yield expressions - existing, substitution
            else:
                exprs_with_counts = list(expressions.items())
                counts = tuple(c for _, c in exprs_with_counts)
                for subset in minimum_integer_vector_iter(counts, minimum):
                    sub_counter = Counter(dict((exprs_with_counts[i][0], c) for i, c in enumerate(subset)))
                    new_substitution = Substitution(substitution)
                    new_substitution[variable] = list(sorted(sub_counter.elements()))
                    yield expressions - sub_counter, new_substitution

        return factory

    @staticmethod
    def _extract_substitution(expression: Expression, pattern: Expression, subst: Substitution) -> bool:
        if isinstance(pattern, Variable):
            if pattern.name in subst:
                if expression != subst[pattern.name]:
                    return False
            else:
                subst[pattern.name] = expression
            return CommutativeMatcher._extract_substitution(expression, pattern.expression, subst)
        elif isinstance(pattern, Operation):
            assert isinstance(expression, type(pattern))
            for expr, patt in zip(expression.operands, pattern.operands):
                if not CommutativeMatcher._extract_substitution(expr, patt, subst):
                    return False
        return True

    @staticmethod
    def _unify_substitutions(*substs: Substitution) -> Substitution:
        unified = substs[0]
        for subst in substs[1:]:
            for variable, value in subst.items():
                if variable in unified:
                    if unified[variable] != value:
                        return None
                else:
                    unified[variable] = value
        return unified


    @staticmethod
    def split_parts(expressions: List[Expression]) -> CommutativeParts:
        constants  = []
        syntactics = []
        variables  = []
        fixed      = []

        for expression in sorted(expressions):
            if expression.is_constant:
                constants.append(expression)
            elif expression.head is None:
                wc = expression
                while isinstance(wc, Variable):
                    wc = wc.expression
                if wc.fixed_size:
                    fixed.append((expression, wc.min_count))
                else:
                    variables.append((expression, wc.min_count))
            elif expression.is_syntactic:
                syntactics.append(expression)
            else:
                fixed.append((expression, 1))

        return CommutativeParts(constants, syntactics, variables, fixed)

    @staticmethod
    def split_expressions(expressions: List[Expression]) -> Tuple[List[Expression], List[Expression]]:
        constants = []
        syntactics = []

        for expression in sorted(expressions):
            if expression.is_syntactic:
                syntactics.append(expression)
            else:
                constants.append(expression)

        return constants, syntactics

def iterator_chain(initial_data, *factories):
    f_count = len(factories)
    iterators = [None] * f_count
    next_data = initial_data
    i = 0
    while True:
        try:
            while i < f_count:
                if iterators[i] is None:
                    iterators[i] = factories[i](*next_data)
                next_data = iterators[i].__next__()
                i += 1
            yield next_data
            i -= 1
        except StopIteration:
            iterators[i] = None
            i -= 1
            if i < 0:
                break


if __name__ == '__main__': # pragma: no cover
    def _main():
        # pylint: disable=invalid-name,bad-continuation
        f = Operation.new('f', Arity.variadic, associative=True, one_identity=True, commutative=True)
        f2 = Operation.new('f2', Arity.variadic, associative=True, one_identity=True, commutative=True)
        g = Operation.new('g', Arity.unary)
        g2 = Operation.new('g2', Arity.unary)

        x_ = Variable.dot('x')
        x__ = Variable.plus('x')
        x___ = Variable.star('x')
        y_ = Variable.dot('y')
        z_ = Variable.dot('z')
        z___ = Variable.star('z')

        a = Symbol('a')
        b = Symbol('b')
        c = Symbol('c')

        patterns = [
            #f(b, x_, g(b), g(x_, b), z___),
            #f(a, x_, g(a), g(x_, a), z___),
            #f(b, y_, g(b), g(x_, b), z___),
            #f(a, y_, g(a), g(x_, a), z___),
            f(x_, y_, g(x_), g(y_), g(x_, y_), z_),
            #f(c, x_, g(y_), g(c)),
            #f(c, x_, g(y_), g(c), g(g(a)), g(a, c)),
            #f(b, c, x_, g(x_), g(x_)),
            #f(g(x_), g(c), b, g(g(y_)))
        ]

        #expr = f(a, b, g(a), g(b))
        expr = f(a, b, g(a), g(b), g(a, b), g(b, a))

        print('Expression: ', expr)

        matcher = CommutativeMatcher()

        parts = [matcher.split_parts(p.operands) for p in patterns]

        for part in parts:
            for op in part.syntactic:
                matcher.add_pattern(op)

        expr_consts, expr_synts = matcher.split_expressions(expr.operands)

        for e in expr_synts:
            matcher.add_expression(e)

        matcher.bipartite.as_graph().render()

        for i, pattern in enumerate(parts):
            print ('-------- %s ----------' % patterns[i])
            for match in matcher.match(expr.operands, pattern, str(patterns[i])):
                print ('match: ', match)

        #print(ManyToOneMatcher._split_parts([a, a, b, g(a), g(b), g(x_, y_), g(x_), f2(y_, a), x_, y_, x__, x___]))

        #matcher = ManyToOneMatcher(*patterns)

        #matcher.match(f(a, b, g(a), g(b)))



    _main()
