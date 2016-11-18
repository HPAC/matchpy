# -*- coding: utf-8 -*-
from typing import (Any, Dict, Generic,  # pylint: disable=unused-import
                    Iterable, Iterator, List, Mapping, NamedTuple, Optional,
                    Set, Tuple, Type, TypeVar, Union, cast)

from multiset import Multiset

from ..constraints import Constraint, MultiConstraint
from ..expressions import (Expression, Operation, Substitution, Symbol,
                           Variable, Wildcard)
from ..utils import (VariableWithCount,
                     fixed_integer_vector_iter, iterator_chain)
from .bipartite import BipartiteGraph, enum_maximum_matchings_iter
from .common import match_operation, match_variable, match_wildcard, CommutativePatternsParts, match_commutative_operation
from .syntactic import DiscriminationNet


class ManyToOneMatcher(object):
    def __init__(self, *patterns: Expression) -> None:
        self.patterns = patterns
        self.graphs = {}  # type: Dict[Type[Operation], Set[Expression]]
        self.commutative = {}  # type: Dict[Type[Operation], CommutativeMatcher]

        for pattern in patterns:
            self._extract_subexpressions(pattern, False, self.graphs)

        for g, exprs in self.graphs.items():
            matcher = CommutativeMatcher(self._match)
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
            for match in self._match([expression], pattern, Substitution()):
                yield pattern, match

    def _match(self, expressions, pattern, subst):
        if isinstance(pattern, Variable):
            yield from match_variable(expressions, pattern, subst, self._match)

        elif isinstance(pattern, Wildcard):
            yield from match_wildcard(expressions, pattern, subst)

        elif isinstance(pattern, Symbol):
            if len(expressions) == 1 and expressions[0] == pattern:
                if pattern.constraint is None or pattern.constraint(subst):
                    yield subst

        else:
            assert isinstance(pattern, Operation), "Unexpected expression of type {!r}".format(type(pattern))
            if len(expressions) != 1 or not isinstance(expressions[0], pattern.__class__):
                return
            op_expr = cast(Operation, expressions[0])

            if op_expr.commutative:
                matcher = self.commutative[type(op_expr)]
                parts = CommutativePatternsParts(type(pattern), *pattern.operands)
                yield from matcher.match(op_expr.operands, parts, subst)
            else:
                for result in match_operation(op_expr.operands, pattern, subst, self._match):
                    if pattern.constraint is None or pattern.constraint(result):
                        yield result

    @staticmethod
    def _extract_subexpressions(expression: Expression, include_constant: bool,
                                subexpressions: Dict[Type[Operation], Set[Expression]]=None) \
            -> Dict[Type[Operation], Set[Expression]]:
        if subexpressions is None:
            subexpressions = {}
        for subexpr, _ in expression.preorder_iter(lambda e: isinstance(e, Operation) and e.commutative):
            operation = cast(Operation, subexpr)
            op_type = cast(Type[Operation], operation.__class__)
            parts = CommutativePatternsParts(op_type, *operation.operands)
            expressions = parts.syntactic
            if include_constant:
                expressions = expressions + parts.constant
            subexpressions.setdefault(op_type, set()).update(expressions)
        return subexpressions


class CommutativeMatcher(object):
    def __init__(self, matcher) -> None:
        self.patterns = set()  # type: Set[Expression]
        self.expressions = set()  # type: Set[Expression]
        self.net = DiscriminationNet()
        self.bipartite = BipartiteGraph()  # type: BipartiteGraph
        self.matcher = matcher

    def add_pattern(self, pattern: Expression) -> None:
        if not pattern.is_syntactic:
            raise ValueError("Can only add syntactic subpatterns.")
        if pattern not in self.patterns:
            self.patterns.add(pattern)
            self.net.add(pattern)

    def add_expression(self, expression: Expression) -> None:
        if not expression.is_constant:
            raise ValueError("The expression must be constant.")
        if expression not in self.expressions:
            self.expressions.add(expression)
            for pattern in self.net.match(expression):
                subst = Substitution()
                if subst.extract_substitution(expression, pattern):
                    self.bipartite[expression, pattern] = subst

    def match(self, expression: List[Expression], pattern: CommutativePatternsParts, substitution: Substitution=None) -> Iterator[Substitution]:
        yield from match_commutative_operation(expression, pattern, substitution, self.matcher, self._syntactic_match)

    def _syntactic_match(self, syntactics, patterns) -> Iterator[Substitution]:
        subgraph = self._build_bipartite(syntactics, patterns)
        match_iter = enum_maximum_matchings_iter(subgraph)
        try:
            matching = next(match_iter)
        except StopIteration:
            return
        if len(matching) < len(patterns):
            return

        if self._is_canonical_matching(matching):
            substitutions = (subgraph[s] for s in matching.items())
            try:
                first_subst = next(substitutions)
                result = first_subst.union(*substitutions)
                matched = Multiset(e for e, _ in matching)
                yield result, syntactics - matched
            except (ValueError, StopIteration):
                pass
        for matching in match_iter:
            if self._is_canonical_matching(matching):
                substitutions = (subgraph[s] for s in matching.items())
                try:
                    first_subst = next(substitutions)
                    result = first_subst.union(*substitutions)
                    matched = Multiset(e for e, _ in matching)
                    yield result, syntactics - matched
                except (ValueError, StopIteration):
                    pass

    def _build_bipartite(self, syntactics: Multiset, patterns: Multiset):
        bipartite = BipartiteGraph()  # type: BipartiteGraph
        for (expr, patt), m in self.bipartite.items():
            for i in range(syntactics[expr]):
                for j in range(patterns[patt]):
                    bipartite[(expr, i), (patt, j)] = m
        return bipartite

    @staticmethod
    def _is_canonical_matching(matching: Dict[Tuple[Tuple[Expression, int], Tuple[Expression, int]], Any]) -> bool:
        for (_, i), (_, j) in matching.items():
            if i > j:
                return False
        return True
