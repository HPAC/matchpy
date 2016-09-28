# -*- coding: utf-8 -*-
import itertools
import operator
from collections import Counter as OriginalCounter
from typing import (Any, Dict, Generic, Iterable, Iterator, List, Mapping, # pylint: disable=unused-import
                    Optional, Set, Tuple, Type, TypeVar, Union, cast)

from patternmatcher.bipartite import (BipartiteGraph,
                                      enum_maximum_matchings_iter)
from patternmatcher.expressions import (Arity, Expression, Operation, Symbol,
                                        Variable, Wildcard)
from patternmatcher.functions import (_match_operation, _match_variable,
                                      _match_wildcard)
from patternmatcher.syntactic import DiscriminationNet
from patternmatcher.utils import (commutative_sequence_variable_partition_iter,
                                  fixed_integer_vector_iter, iterator_chain,
                                  minimum_integer_vector_iter)

T = TypeVar('T')
class Counter(OriginalCounter, Mapping[T, int], Generic[T]):
    # pylint: disable=abstract-method
    def __le__(self, other: OriginalCounter):
        '''Checks if all counts from this counter are less than or equal to the other.

        >>> Counter('ab') <= Counter('aabc')
        True

        '''
        if not isinstance(other, OriginalCounter):
            return NotImplemented
        for elem in self:
            if self[elem] > other[elem]:
                return False
        for elem in other:
            if self[elem] > other[elem]:
                return False
        return True

    def __ge__(self, other: OriginalCounter):
        '''Checks if all counts from this counter are greater than or equal to the other.

        >>> Counter('aabc') >= Counter('ab')
        True

        '''
        if not isinstance(other, OriginalCounter):
            return NotImplemented
        for elem in self:
            if self[elem] < other[elem]:
                return False
        for elem in other:
            if self[elem] < other[elem]:
                return False
        return True


class CommutativePatternsParts(object):
    """Representation of the parts of a commutative pattern expression.

    This data structure contains all the operands of a commutative operation pattern.
    They are distinguished by how they need to be matched against an expression.

    All parts are represented by a :class:`collections.Counter`. This is essentially
    equivalent to a multiset. It is used, because the order of operands does not matter
    in a commutative operation. The count (value) represents how many times the expression (key)
    occured in the operation.

    In addition, some lengths are precalculated during the initialization of this data structure
    so that they do not have to be recalculated later.

    This data structure is meant to be immutable, so do not change any of its attributes!

    Attributes:
        operation (type):
            The type of of the original pattern expression. Must be a subclass of
            :class:`.Operation`.

        constant (collections.Counter):
            A :class:`collections.Counter` representing the constant operands of the pattern.
            An expression is constant, if it does not contain variables or wildcards.
        syntactic (collections.Counter):
            A :class:`collections.Counter` representing the syntactic operands of the pattern.
            An expression is syntactic, if it does contain neither associative nor commutative operations
            nor sequence variables. Here, constant expressions and variables also get their own counters,
            so they are not included in this counter.
        sequence_variables (collections.Counter):
            A :class:`collections.Counter` representing the sequence variables of the pattern.
            Here the key is a tuple of the form `(name, min_count)` where `name` is the name of the
            sequence variable and `min_count` is the minimum length of the sequence variable.
            For wildcards without variable, the name will be `None`.
        fixed_variables (collections.Counter):
            A :class:`collections.Counter` representing the fixed length variables of the pattern.
            Here the key is a tuple of the form `(name, length)` of the variable.
            For wildcards without variable, the name will be `None`.
        rest (collections.Counter):
            A :class:`collections.Counter` representing the operands of the pattern that do not fall
            into one of the previous categories. That means it contains operation expressions, which
            are not syntactic.

        length (int):
            The total count of operands of the commutative operation pattern.
        sequence_variable_min_length (int):
            The total combined minimum length of all sequence variables in the commutative
            operation pattern. This is the sum of the `min_count` attributes of the sequence
            variables.
        fixed_variable_length (int):
            The total combined length of all fixed length variables in the commutative
            operation pattern. This is the sum of the `min_count` attributes of the
            variables.
        rest_length (int):
            The total length of the `rest` part i.e. the sum of its counts.
        constant_length (int):
            The total length of the `constant` part i.e. the sum of its counts.
        syntactic_length (int):
            The total length of the `syntactic` part i.e. the sum of its counts.
    """
    def __init__(self, operation: Type[Operation], *expressions: Expression) -> None:
        self.operation = operation
        self.length = len(expressions)

        self.constant = Counter() # type: Counter[Expression]
        self.syntactic = Counter() # type: Counter[Expression]
        self.sequence_variables = Counter() # type: Counter[Tuple[str, int]]
        self.fixed_variables = Counter() # type: Counter[Tuple[str, int]]
        self.rest = Counter() # type: Counter[Expression]

        self.sequence_variable_min_length = 0
        self.fixed_variable_length = 0

        for expression in sorted(expressions):
            if expression.is_constant:
                self.constant[expression] += 1
            elif expression.head is None:
                wc = cast(Wildcard, expression)
                name = None # type: Optional[str]
                if isinstance(wc, Variable):
                    name = wc.name
                    wc = cast(Wildcard, wc.expression)
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
    """Special `dict` for substitutions with nicer formatting.

    The key is a variable's name and the value the replacement for it.
    """

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
        self.graphs = {} # type: Dict[Type[Operation], Set[Expression]]
        self.commutative = {} # type: Dict[Type[Operation], CommutativeMatcher]

        for pattern in patterns:
            self._extract_subexpressions(pattern, False, self.graphs)

        for g, exprs in self.graphs.items():
            matcher = CommutativeMatcher(self)
            for expr in exprs:
                matcher.add_pattern(expr)
            self.commutative[g] = matcher

    def match(self, expression):
        subexpressions = self._extract_subexpressions(expression, True)
        for t, es in subexpressions.items():
            if t in self.commutative:
                for e in es:
                    self.commutative[t].add_expression(e)

        for pattern in self.patterns:
            for match in self._match(expression, pattern, Substitution()):
                yield pattern, match 

    def _match(self, expression, pattern, subst):
        if isinstance(expression, list):
            if len(expression) == 1:
                expression = expression[0]
            else:
                # TODO: Correct behaviour
                return

        if isinstance(pattern, Variable):
            yield from _match_variable([expression], pattern, subst, self._match)

        elif isinstance(pattern, Wildcard):
            yield from _match_wildcard([expression], pattern, subst)

        elif isinstance(pattern, Symbol):
            if expression == pattern:
                if pattern.constraint is None or pattern.constraint(subst):
                    yield subst

        else:
            assert isinstance(pattern, Operation), 'Unexpected expression of type %r' % type(pattern)
            if not isinstance(expression, pattern.__class__):
                return
            op_expr = cast(Operation, expression)

            if op_expr.commutative:
                matcher = self.commutative[type(op_expr)]
                parts = CommutativePatternsParts(type(pattern), *pattern.operands)
                yield from matcher.match(op_expr.operands, parts)
            else:
                for result in _match_operation(op_expr.operands, pattern, subst, self._match):
                    if pattern.constraint is None or pattern.constraint(result):
                        yield result

    @staticmethod
    def _extract_subexpressions(expression: Expression, include_constant: bool, subexpressions:Dict[Type[Operation], Set[Expression]]=None) -> Dict[Type[Operation], Set[Expression]]:
        if subexpressions is None:
            subexpressions = {}
        for subexpr, _ in expression.preorder_iter(lambda e: isinstance(e, Operation) and e.commutative):
            operation = cast(Operation, subexpr)
            op_type = cast(Type[Operation], operation.__class__)
            if op_type not in subexpressions:
                subexpressions[op_type] = set()
            parts = CommutativePatternsParts(op_type, *operation.operands)
            expressions = parts.syntactic
            if include_constant:
                expressions = expressions + parts.constant
            subexpressions[op_type].update(expressions)
        return subexpressions

class CommutativeMatcher(object):
    _cnt = 0

    def __init__(self, parent: ManyToOneMatcher) -> None:
        self.patterns = set() # type: Set[Expression]
        self.expressions = set() # type: Set[Expression]
        self.net = DiscriminationNet()
        self.bipartite = BipartiteGraph() # type: BipartiteGraph
        self.parent = parent

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

    def match(self, expression: List[Expression], pattern: CommutativePatternsParts) -> Iterator[Substitution]:
        if any(not e.is_constant for e in expression):
            raise ValueError('All given expressions must be constant.')

        expressions = Counter(expression) # type: Counter[Expression]

        if not (pattern.constant <= expressions):
            return

        expressions.subtract(pattern.constant)

        rest, syntactics = self.split_expressions(expressions)

        syn_patt_count = sum(pattern.syntactic.values())

        if syn_patt_count > sum(syntactics.values()):
            return

        if pattern.syntactic:
            subgraph = self._build_bipartite(syntactics, pattern.syntactic)
            #subgraph.as_graph().render('tmp/' + label + '.gv')
            match_iter = enum_maximum_matchings_iter(subgraph)
            try:
                matching = next(match_iter)
            except StopIteration:
                return
            if len(matching) < syn_patt_count:
                return

            if self._is_canonical_matching(matching):
                subst = self._unify_substitutions(*(subgraph[s] for s in matching.items()))
                matched = Counter(e for e, _ in matching) # type: Counter[Expression]
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
        else:
            yield from self._matches_from_matching(Substitution(), expressions, pattern)

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
        bipartite = BipartiteGraph() # type: BipartiteGraph
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

        fixed_vars = Counter(pattern.fixed_variables) # type: Counter[Tuple[str, int]]
        for (name, length), count in pattern.fixed_variables.items():
            if name in subst:
                if pattern.operation.associative and isinstance(subst[name], pattern.operation):
                    needed_count = Counter(cast(Operation, subst[name]).operands) # type: Counter[Expression]
                elif isinstance(subst[name], (list, tuple)):
                    needed_count = Counter(cast(Iterable[Expression], subst[name]))
                else:
                    needed_count = Counter({subst[name]: 1})
                if count > 1:
                    for k in needed_count:
                        needed_count[k] = needed_count[k] * count
                if not (needed_count <= remaining):
                    return
                remaining -= needed_count
                del fixed_vars[(name, length)]

        factories = [self._fixed_expr_factory(e) for e in pattern.rest]
        
        if not pattern.operation.associative:
            factories += [self._fixed_var_iter_factory(v, l, c) for (v, l), c in fixed_vars.items()]

        expr_counter = Counter(remaining) # type: Counter[Expression]

        for rem_expr, subst in iterator_chain((expr_counter, subst), *factories):
            sequence_vars = pattern.sequence_variables
            if pattern.operation.associative:
                sequence_vars = sequence_vars + fixed_vars
            for sequence_subst in commutative_sequence_variable_partition_iter(Counter(rem_expr), sequence_vars):
                s = Substitution((var, sorted(exprs.elements())) for var, exprs in sequence_subst.items())
                if pattern.operation.associative:
                    for v, l in fixed_vars:
                        value = cast(list, v)
                        if len(value) > l:
                            s[v] = pattern.operation(*value)
                        elif l == len(value) and l == 1:
                            s[v] = value[0]
                result = self._unify_substitutions(subst, s)
                if result is not None:
                    yield result

    def _fixed_expr_factory(self, expression):
        def factory(expressions, substitution):
            for expr in expressions:
                if expr.head == expression.head:
                    for subst in self.parent._match(expr, expression, substitution):
                        yield expressions - Counter({expr: 1}), subst

        return factory

    @staticmethod
    def _fixed_var_iter_factory(variable, length, count):
        def factory(expressions, substitution):
            if variable in substitution:
                existing = Counter(not isinstance(substitution[variable], list) and [substitution[variable]] or substitution[variable]) * count
                if not (existing <= expressions):
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
                if not (existing <= expressions):
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
            op_expression = cast(Operation, expression)
            for expr, patt in zip(op_expression.operands, pattern.operands):
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
    def split_expressions(expressions: Counter[Expression]) -> Tuple[Counter[Expression], Counter[Expression]]:
        constants = Counter() # type: Counter[Expression]
        syntactics = Counter() # type: Counter[Expression]

        for expression, count in expressions.items():
            if expression.is_syntactic or not (isinstance(expression, Operation) and (expression.associative or expression.commutative)):
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
            f(y_, y_, g(x_), g(x_)),
            f(x_, g(f(x_, z___)))
        ]

        #expr = f(a, b, g(a), g(b))
        expr = f(a, b, g(a), g(b), g(a, b), g(b, a))
        #expr = f(g(a), g(a), g(b), g(b))
        #expr = f(a, b, g(f(a, a, b)))

        print('Expression: ', expr)

        matcher = ManyToOneMatcher(*patterns)
        #matcher = CommutativeMatcher(parent)

        #parts = [CommutativePatternsParts(type(p), *p.operands) for p in patterns]

        #for part in parts:
        #    for op in part.syntactic:
        #        matcher.add_pattern(op)

        #_, expr_synts = matcher.split_expressions(Counter(expr.operands))

        #for e in expr_synts:
        #    matcher.add_expression(e)

        #matcher.bipartite.as_graph().render('tmp/BP.gv')

        for pattern, matches in itertools.groupby(matcher.match(expr), operator.itemgetter(0)):
            print ('-------- %s ----------' % pattern)
            for _, match in matches:
                print ('match: ', match)

        #for i, pattern in enumerate(parts):
        #    print ('-------- %s ----------' % patterns[i])
        #    for match in matcher.match(expr.operands, pattern):
        #        print ('match: ', match)



    _main()
