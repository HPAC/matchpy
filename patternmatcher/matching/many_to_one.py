# -*- coding: utf-8 -*-
from typing import (Any, Dict, FrozenSet, Iterator, List, Sequence, Set, Tuple,
                    Type, cast)

from multiset import Multiset

from ..expressions import (Expression, FrozenExpression, Operation,
                           Substitution, Symbol, Variable, Wildcard, freeze)
from .bipartite import BipartiteGraph, enum_maximum_matchings_iter
from .common import (CommutativePatternsParts, match_commutative_operation,
                     match_operation, match_variable, match_wildcard, Matcher)
from .syntactic import DiscriminationNet


class ManyToOneMatcher(object):
    r"""Pattern matcher that matches a set of patterns against a subject.

    It does so more efficiently than using one-to-one matching for each pattern individually
    by reusing structural similarities of the patterns.

    Attributes:
        patterns (FrozenSet[FrozenExpression]):
            The set of patterns that the matcher uses.
        commutative_matchers (Dict[Type[Operation], CommutativeMatcher]):
            A dictionary holding a :class:`CommutativeMatcher` instance for every type of commutative operation
            occurring in the patterns.
    """
    def __init__(self, *patterns: Expression) -> None:
        """Create a new many-to-one matcher.

        Note that the patterns cannot be modified after the instance has been created.
        In case you want to change the set of patterns, you have to create a new matcher instance.

        Args:
            *patterns:
                The patterns that are used for matching.
                Note that the patterns will be :term:`frozen` and hence cannot be changed after the creation of
                the matcher.
        """
        self.patterns = frozenset(freeze(pattern) for pattern in patterns)
        self.commutative_matchers = {}  # type: Dict[Type[Operation], CommutativeMatcher]
        subexpressions = {}  # type: Dict[Type[Operation], Set[FrozenExpression]]

        for pattern in self.patterns:
            self._extract_subexpressions(pattern, False, subexpressions)

        for operation_type, operands in subexpressions.items():
            matcher = CommutativeMatcher(self._match)
            for operand in operands:
                matcher.add_pattern(operand)
            self.commutative_matchers[operation_type] = matcher

    def match(self, expression: Expression) -> Iterator[Tuple[FrozenExpression, Substitution]]:
        """Match the given expression against each pattern.

        The expression will be :term:`frozen` because parts of it might be used as dictionary keys during matching.

        Example:

            >>> pattern1 = freeze(f(a, x_))
            >>> pattern2 = freeze(f(y_, b))
            >>> matcher = ManyToOneMatcher(pattern1, pattern2)
            >>> for pattern, match in sorted(matcher.match(freeze(f(a, b)))):
            ...     print(pattern, ':', match)
            f(a, x_) : x ← b
            f(y_, b) : y ← a

        Args:
            expression: The expression to match.

        Yields:
            For every match, the matching pattern and substitution is yielded as a tuple.
            Note that the pattern will be a :term:`frozen` version of the original pattern., i.e. it will be
            equivalent to the original pattern but might not be identical.
        """
        subexpressions = self._extract_subexpressions(freeze(expression), True)
        for t, es in subexpressions.items():
            if t in self.commutative_matchers:
                for e in es:
                    self.commutative_matchers[t].add_expression(e)

        for pattern in self.patterns:
            for match in self._match([expression], pattern, Substitution()):
                yield pattern, match

    def _match(self, expressions: Sequence[FrozenExpression], pattern: FrozenExpression, subst: Substitution) \
        -> Iterator[Substitution]:
        """Delegates the matching of the expressions depending on the type of the pattern.

        Compared to the regular one-to-one matching, it uses the CommutativeMatcher instances in
        :attr:`commutative_matchers` for commutative expressions.
        """
        if isinstance(pattern, Variable):
            yield from match_variable(expressions, pattern, subst, self._match)

        elif isinstance(pattern, Wildcard):
            yield from match_wildcard(expressions, pattern, subst)

        elif isinstance(pattern, Symbol):
            if len(expressions) == 1 and isinstance(expressions[0], type(pattern)) and expressions[0].name == pattern.name:
                if pattern.constraint is None or pattern.constraint(subst):
                    yield subst

        else:
            assert isinstance(pattern, Operation), "Unexpected expression of type {!r}".format(type(pattern))
            if len(expressions) != 1 or not isinstance(expressions[0], pattern.__class__):
                return
            op_expr = cast(Operation, expressions[0])

            if op_expr.commutative:
                matcher = self.commutative_matchers[type(op_expr)]
                parts = CommutativePatternsParts(type(pattern), *pattern.operands)
                yield from matcher.match(op_expr.operands, parts, subst)
            else:
                for result in match_operation(op_expr.operands, pattern, subst, self._match):
                    if pattern.constraint is None or pattern.constraint(result):
                        yield result

    @staticmethod
    def _extract_subexpressions(expression: FrozenExpression, include_constant: bool,
                                subexpressions: Dict[Type[Operation], Set[FrozenExpression]]=None) \
            -> Dict[Type[Operation], Set[FrozenExpression]]:
        """Extracts all syntactic subexpressions that are operands of a commutative subexpression.

        Args:
            expression:
                The expression from which the subexpressions are extracted.
            include_constant:
                Iff True, constant syntactic subexpressions are extracted. Otherwise, they will be excluded.
            subexpressions:
                If given, it is used to store the subexpressions, otherwise a new dictionary is created.

        Returns:
            The subexpressions grouped by the type of the commutative expression they occur in as a dictionary.
            The key is the commutative operation and the value a set of expressions that occurs.
        """
        if subexpressions is None:
            subexpressions = {}
        for subexpr, _ in expression.preorder_iter(lambda e: isinstance(e, Operation) and e.commutative):
            operation = cast(Operation, subexpr)
            op_type = cast(Type[Operation], operation.__class__)
            expressions = set()
            for operand in operation.operands:
                if operand.is_syntactic and (not operand.is_constant or include_constant):
                    expressions.add(operand)
            subexpressions.setdefault(op_type, set()).update(expressions)
        return subexpressions


class CommutativeMatcher(object):
    r"""A matcher for commutative patterns.

    Creates a :class:`.DiscriminationNet` from the patterns and uses it to populate a
    :class:`.BipartiteGraph` connecting patterns and matching expressions. This graph is then
    used for faster many-to-one matching.

    All patterns have to be added via :meth:`add_pattern` first, then all expressions have to be added
    via :meth:`add_expression`. Finally, expressions can be matched with :meth:`match`.

    Attributes:
        patterns (Set[FrozenExpression]):
            A set of the patterns that have been added to the matcher. Every pattern is :term:`syntactic`.
            Patterns can be added via :meth:`add_pattern`.
        expressions (Set[FrozenExpression]):
            A set of the expressions that have been added to the matcher. Every expression is :term:`syntactic` and
            :term:`constant`. Expressions can be added via :meth:`add_expression`.
        net (DiscriminationNet[FrozenExpression]):
            A discrimination net constructed from the pattern set.
        bipartite (BipartiteGraph[FrozenExpression, FrozenExpression, Substitution]):
            A bipartite graph connecting expressions and patterns. An edge represents a match between the expression
            and pattern and is labeled with the match substitution.
        matcher (Matcher):
            The matching function that is used for recursive matching, i.e. for every non-syntactic expression.

    """
    def __init__(self, matcher: Matcher) -> None:
        self.patterns = set()  # type: Set[FrozenExpression]
        self.expressions = set()  # type: Set[FrozenExpression]
        self.net = DiscriminationNet()
        self.bipartite = BipartiteGraph()  # type: BipartiteGraph
        self.matcher = matcher

    def add_pattern(self, pattern: FrozenExpression) -> None:
        if not pattern.is_syntactic:
            raise ValueError("Can only add syntactic subpatterns.")
        if pattern not in self.patterns:
            self.patterns.add(pattern)
            self.net.add(pattern)

    def add_expression(self, expression: FrozenExpression) -> None:
        if not expression.is_constant or not expression.is_syntactic:
            raise ValueError("The expression must be syntactic and constant.")
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
