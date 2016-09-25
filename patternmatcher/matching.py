# -*- coding: utf-8 -*-

from collections import Counter
from typing import Dict, Set, cast, Tuple, Union, List, Iterator, Any

from patternmatcher.bipartite import (BipartiteGraph,
                                      enum_maximum_matchings_iter)
from patternmatcher.expressions import (Arity, Expression, Operation, Symbol,
                                        Variable)
from patternmatcher.syntactic import DiscriminationNet
from patternmatcher.utils import fixed_integer_vector_iter, minimum_integer_vector_iter, iterator_chain, commutative_sequence_variable_partition_iter

class CommutativePatternsParts(object):
    def __init__(self, operation: type, *expressions: Expression) -> None:
        self.operation = operation
        self.length = len(expressions)

        self.constant = Counter()
        self.syntactic = Counter()
        self.sequence_variables = Counter()
        self.fixed_variables = Counter()
        self.rest = Counter()

        self.sequence_variable_min_length = 0
        self.fixed_variable_length = 0

        for expression in sorted(expressions):
            if expression.is_constant:
                self.constant[expression] += 1
            elif expression.head is None:
                wc = expression
                name = None
                if isinstance(wc, Variable):
                    name = wc.name
                    wc = wc.expression
                if wc.fixed_size:
                    self.fixed_variables[(name, wc.min_count)] += 1
                    self.fixed_variable_length += wc.min_count
                else:
                    self.sequence_variables[(name, wc.min_count)] += 1
                    self.sequence_variable_min_length += wc.min_count
            elif expression.is_syntactic:
                self.syntactic[expression] += 1
            else:
                self.rest[expression] += 1

        self.rest_length = sum(self.rest.values())
        self.constant_length = sum(self.rest.values())
        self.syntactic_length = sum(self.syntactic.values())


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
                bipartites[t].as_graph().render('tmp/' + t.__name__ + '.gv')

    @staticmethod
    def _extract_subexpressions(expression: Expression, include_constant: bool, subexpressions:Dict[type, Set[Expression]]=None) -> Dict[type, Set[Expression]]:
        if subexpressions is None:
            subexpressions = {}
        for subexpr, _ in expression.preorder_iter(lambda e: isinstance(e, Operation) and e.commutative):
            operation = cast(Operation, subexpr)
            if type(operation) not in subexpressions:
                subexpressions[type(operation)] = set()
            parts = CommutativePatternsParts(type(operation), *operation.operands)
            expressions = parts.syntactic
            if include_constant:
                expressions += parts.constant
            subexpressions[type(operation)].update(expressions)
        return subexpressions

class CommutativeMatcher(object):
    _cnt = 0

    def __init__(self) -> None:
        self.patterns = set() # type: Set[Expression]
        self.expressions = set() # type: Set[Expression]
        self.net = DiscriminationNet()
        self.bipartite = BipartiteGraph()

    def add_pattern(self, pattern: Expression):
        if not pattern.is_syntactic:
            raise ValueError('Can only add syntactic subpatterns.')
        if pattern not in self.patterns:
            self.patterns.add(pattern)
            self.net.add(pattern)

    def add_expression(self, expression: Expression):
        if not expression.is_constant:
            raise ValueError('The expression must be constant.')
        if expression not in self.expressions:
            self.expressions.add(expression)
            for pattern in self.net.match(expression):
                subst = Substitution()
                if self._extract_substitution(expression, pattern, subst):
                    self.bipartite[expression, pattern] = subst

    def match(self, expression: List[Expression], pattern: CommutativePatternsParts, label) -> Iterator[Substitution]:
        if any(not e.is_constant for e in expression):
            raise ValueError('All given expressions must be constant.')

        expressions = Counter(expression)

        if pattern.constant - expressions:
            return

        expressions.subtract(pattern.constant)

        rest, syntactics = self.split_expressions(expressions)

        syn_patt_count = sum(pattern.syntactic.values())

        if syn_patt_count > sum(syntactics.values()):
            return

        if pattern.syntactic:
            subgraph = self._build_bipartite(syntactics, pattern.syntactic)
            subgraph.as_graph().render('tmp/' + label + '.gv')
            match_iter = enum_maximum_matchings_iter(subgraph)
            try:
                matching = next(match_iter)
            except StopIteration:
                return
            if len(matching) < syn_patt_count:
                return

            if self._is_canonical_matching(matching):
                subst = self._unify_substitutions(*(subgraph[s] for s in matching.items()))
                matched = Counter(e for e, _ in matching)
                remaining = rest + (syntactics - matched)
                if subst is not None:
                    yield from self._matches_from_matching(subst, remaining, pattern)
            for matching in match_iter:                
                if self._is_canonical_matching(matching):
                    subst = self._unify_substitutions(*(subgraph[s] for s in matching.items()))
                    matched = Counter(e for e, _ in matching)
                    remaining = rest + (syntactics - matched)
                    if subst is not None:
                        yield from self._matches_from_matching(subst, remaining, pattern)

    def _match_and_remove_constants(self, expressions, pattern):
        constants = sorted(expressions)
        i = 0
        for const_pattern in pattern.constant:
            while constants[i] < const_pattern:
                i += 1
            if constants[i] == const_pattern:
                del constants[i]
            else:
                return
        return constants

    def _build_bipartite(self, syntactics: Counter, patterns: Counter):
        bipartite = BipartiteGraph()
        for (expr, patt), m in self.bipartite.items():
            for i in range(syntactics[expr]):
                for j in range(patterns[patt]):
                    bipartite[(expr, i), (patt, j)] = m
        bipartite.as_graph().render('tmp/' + str(CommutativeMatcher._cnt) + '.gv')
        CommutativeMatcher._cnt += 1
        return bipartite

    @staticmethod
    def _is_canonical_matching(matching: Dict[Tuple[Tuple[Expression, int], Tuple[Expression, int]], Any]) -> bool:
        for (_, i), (_, j) in matching.items():
            if i > j:
                return False
        return True

    def _matches_from_matching(self, subst: Substitution, remaining: Counter, pattern: CommutativePatternsParts) -> Iterator[Substitution]:
        needed_length = pattern.sequence_variable_min_length + pattern.fixed_variable_length + pattern.rest_length

        if sum(remaining.values()) < needed_length:
            return

        fixed_vars = Counter(pattern.fixed_variables)
        for (name, length), count in pattern.fixed_variables.items():
            if name in subst:
                if pattern.operation.associative and isinstance(subst[name], pattern.operation):
                    needed_count = Counter(subst[name].operands)
                    if count > 1:
                        for k in needed_count:
                            needed_count[k] = needed_count[k] * count
                    if needed_count - remaining:
                        return
                    remaining -= needed_count
                else:
                    if remaining[subst[name]] < count:
                        return
                    remaining[subst[name]] -= count
                del fixed_vars[(name, length)]

        factories = [self._fixed_expr_factory(e) for e in pattern.rest]
        
        if not pattern.operation.associative:
            factories += [self._fixed_var_iter_factory(v, l, c) for (v, l), c in fixed_vars.items()]

        expr_counter = Counter(remaining)

        for rem_expr, subst in iterator_chain((expr_counter, subst), *factories):
            sequence_vars = pattern.sequence_variables
            if pattern.operation.associative:
                sequence_vars += fixed_vars
            for sequence_subst in commutative_sequence_variable_partition_iter(Counter(rem_expr), sequence_vars):
                s = Substitution((var, sorted(exprs.elements())) for var, exprs in sequence_subst.items())
                if pattern.operation.associative:
                    for v, l in fixed_vars:
                        if len(s[v]) > l:
                            s[v] = pattern.operation(*s[v])
                        elif l == len(s[v]) and l == 1:
                            s[v] = s[v][0]
                result = self._unify_substitutions(subst, s)
                if result is not None:
                    yield result

    @staticmethod
    def _fixed_expr_factory(expression):
        def factory(expressions, substitution):
            for expr in expressions:
                if expr.head == expression.head:
                    yield expressions - Counter({expr: 1}), substitution

        return factory

    @staticmethod
    def _fixed_var_iter_factory(variable, length, count):
        def factory(expressions, substitution):
            if variable in substitution:
                existing = Counter(not isinstance(substitution[variable], list) and [substitution[variable]] or substitution[variable]) * count
                if existing - expressions:
                    return
                yield expressions - existing, substitution
            else:
                if length == 1:
                    for expr, expr_count in expressions.most_common():
                        if expr_count < count:
                            break
                        new_substitution = Substitution(substitution)
                        new_substitution[variable] = expr
                        yield expressions - Counter({expr: count}), new_substitution
                else:
                    exprs_with_counts = list(expressions.items())
                    counts = tuple(c // count for _, c in exprs_with_counts)
                    for subset in fixed_integer_vector_iter(counts, length):
                        sub_counter = Counter(dict((exprs_with_counts[i][0], c * count) for i, c in enumerate(subset)))
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
        unified = Substitution(substs[0])
        for subst in substs[1:]:
            for variable, value in subst.items():
                if variable in unified:
                    if unified[variable] != value:
                        return None
                else:
                    unified[variable] = value
        return unified

    @staticmethod
    def split_expressions(expressions: Counter) -> Tuple[Counter, Counter]:
        constants = Counter()
        syntactics = Counter()

        for expression, count in expressions.items():
            if expression.is_syntactic:
                syntactics[expression] = count
            else:
                constants[expression] = count

        return constants, syntactics


if __name__ == '__main__': # pragma: no cover
    def _main():
        # pylint: disable=invalid-name,bad-continuation
        f = Operation.new('f', Arity.variadic, associative=True, one_identity=True, commutative=True)
        g = Operation.new('g', Arity.unary)

        x_ = Variable.dot('x')
        y_ = Variable.dot('y')
        z_ = Variable.dot('z')
        z___ = Variable.star('z')

        a = Symbol('a')
        b = Symbol('b')
        c = Symbol('c')

        patterns = [
            f(b, x_, g(b), g(x_, b), z___),
            f(a, x_, g(a), g(x_, a), z___),
            f(b, y_, g(b), g(x_, b), z___),
            f(a, y_, g(a), g(x_, a), z___),
            f(x_, y_, g(x_), g(y_), g(x_, y_), z_),
            f(c, x_, g(y_), g(c)),
            f(c, x_, g(y_), g(c), g(g(a)), g(a, c)),
            f(b, c, x_, g(x_), g(x_)),
            f(g(x_), g(c), b, g(g(y_))),
            f(y_, y_, g(x_), g(x_))
        ]

        #expr = f(a, b, g(a), g(b))
        expr = f(a, b, g(a), g(b), g(a, b), g(b, a))
        #expr = f(g(a), g(a), g(b), g(b))

        print('Expression: ', expr)

        matcher = CommutativeMatcher()

        parts = [CommutativePatternsParts(type(p), *p.operands) for p in patterns]

        for part in parts:
            for op in part.syntactic:
                matcher.add_pattern(op)

        _, expr_synts = matcher.split_expressions(Counter(expr.operands))

        for e in expr_synts:
            matcher.add_expression(e)

        matcher.bipartite.as_graph().render('tmp/BP.gv')

        for i, pattern in enumerate(parts):
            print ('-------- %s ----------' % patterns[i])
            for match in matcher.match(expr.operands, pattern, str(patterns[i])):
                print ('match: ', match)



    _main()
