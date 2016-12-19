# -*- coding: utf-8 -*-
"""Contains the :class:`ManyToOneMatcher` and other classes related to many-to-one pattern matching."""

from typing import (Any, Dict, FrozenSet, Iterator, List, Sequence, Set, Tuple, Type, cast)

from multiset import Multiset

from ..expressions import (Expression, FrozenExpression, Operation, Substitution, Symbol, Variable, Wildcard, freeze)
from .bipartite import BipartiteGraph, enum_maximum_matchings_iter
from .common import (
    CommutativePatternsParts, match_commutative_operation, _non_commutative_match, match_variable, match_wildcard,
    Matcher
)
from .syntactic import DiscriminationNet

__all__ = ['ManyToOneMatcher', 'CommutativeMatcher']


class ManyToOneMatcher(object):
    """Pattern matcher that matches a set of patterns against a subject.

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

            >>> pattern1 = f(a, x_)
            >>> pattern2 = f(y_, b)
            >>> matcher = ManyToOneMatcher(pattern1, pattern2)
            >>> subject = f(a, b)
            >>> for pattern, match in sorted(matcher.match(subject)):
            ...     print(pattern, ':', match)
            f(a, x_) : {x ↦ b}
            f(y_, b) : {y ↦ a}

        Args:
            expression: The expression to match.

        Yields:
            For every match, the matching pattern and substitution is yielded as a tuple.
            Note that the pattern will be a :term:`frozen` version of the original pattern., i.e. it will be
            equivalent to the original pattern but might not be identical.
        """
        expression = freeze(expression)
        subexpressions = self._extract_subexpressions(expression, True)
        for t, es in subexpressions.items():
            if t in self.commutative_matchers:
                for e in es:
                    self.commutative_matchers[t].add_expression(e)

        for pattern in self.patterns:
            for match in self._match([expression], pattern, Substitution()):
                yield pattern, match

    def _match(self, expressions: Sequence[FrozenExpression], pattern: FrozenExpression,
               subst: Substitution) -> Iterator[Substitution]:
        """Delegates the matching of the expressions depending on the type of the pattern.

        Compared to the regular one-to-one matching, it uses the CommutativeMatcher instances in
        :attr:`commutative_matchers` for commutative expressions.
        """
        if isinstance(pattern, Variable):
            yield from match_variable(expressions, pattern, subst, self._match)

        elif isinstance(pattern, Wildcard):
            yield from match_wildcard(expressions, pattern, subst)

        elif isinstance(pattern, Symbol):
            if (
                len(expressions) == 1 and isinstance(expressions[0], type(pattern)) and
                expressions[0].name == pattern.name
            ):
                if pattern.constraint is None or pattern.constraint(subst):
                    yield subst

        else:
            assert isinstance(pattern, Operation), "Unexpected expression of type {!r}".format(type(pattern))
            if len(expressions) != 1 or not isinstance(expressions[0], pattern.__class__):
                return
            op_expr = cast(Operation, expressions[0])
            if not op_expr.symbols >= pattern.symbols:
                return

            if op_expr.commutative:
                matcher = self.commutative_matchers[type(op_expr)]
                parts = CommutativePatternsParts(type(pattern), *pattern.operands)
                for result in matcher.match(op_expr.operands, parts, subst):
                    if pattern.constraint is None or pattern.constraint(result):
                        yield result
            else:
                for result in _non_commutative_match(op_expr.operands, pattern, subst, self._match):
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


Subgraph = BipartiteGraph[Tuple[FrozenExpression, int], Tuple[FrozenExpression, int], Substitution]
SubgraphMatching = Dict[Tuple[Tuple[Expression, int], Tuple[Expression, int]], Substitution]


class CommutativeMatcher(object):
    """A matcher for commutative patterns.

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
            and pattern and is labeled with the match substitution. This graph contains all the subexpressions and
            subpatterns from :attr:`expressions` and :attr:`patterns`.
        matcher (Matcher):
            The matching function that is used for recursive matching, i.e. for every non-syntactic expression.

    """

    def __init__(self, matcher: Matcher) -> None:
        """Create a CommutativeMatcher instance.

        Args:
            matcher: The parent matcher that recursive matching for non-:term:`syntactic` expressions is delegated to.
        """
        self.patterns = set()  # type: Set[FrozenExpression]
        self.expressions = set()  # type: Set[FrozenExpression]
        self.net = DiscriminationNet()
        self.bipartite = BipartiteGraph()  # type: BipartiteGraph
        self.matcher = matcher

    def add_pattern(self, pattern: FrozenExpression) -> None:
        """Add a new syntactic pattern to be matched later.

        Args:
            pattern: The pattern to add.

        Raises:
            ValueError: If the given pattern is not syntactic.
        """
        if not pattern.is_syntactic:
            raise ValueError("Can only add syntactic subpatterns.")
        if pattern not in self.patterns:
            self.patterns.add(pattern)
            self.net.add(pattern)

    def add_expression(self, expression: FrozenExpression) -> None:
        """Add a new constant syntactic expression to be matched later.

        Args:
            expression: The expression to add.

        Raises:
            ValueError: If the given expression is not syntactic or not constant.
        """
        if not expression.is_constant or not expression.is_syntactic:
            raise ValueError("The expression must be syntactic and constant.")
        if expression not in self.expressions:
            self.expressions.add(expression)
            for pattern in self.net.match(expression):
                subst = Substitution()
                if subst.extract_substitution(expression, pattern):
                    self.bipartite[expression, pattern] = subst

    def match(self, expression: List[Expression], pattern: CommutativePatternsParts, substitution: Substitution=None) \
            -> Iterator[Substitution]:
        """Match the expression against the pattern and yield each valid substitution.

        Uses :func:`.match_commutative_operation` with this matcher's parent :attr:`matcher` to recursively match
        non-:term:`syntactic` expressions. All syntactic expressions are first tried to match using the matcher's
        :attr:`bipartite` graph.

        Args:
            expression:
                A list of operands of the commutative operation expression to match.
            pattern:
                The operands of the commutative operation pattern to match.
            substitution:
                The initial substitution as it might already be filled by previous matching steps.
                If not given, an empty substitution is used instead.

        Yields:
            Each substitution that is a valid match for the pattern and expression.
        """
        yield from match_commutative_operation(expression, pattern, substitution, self.matcher, self._syntactic_match)

    def _syntactic_match(self, expressions: Multiset[FrozenExpression], patterns: Multiset[FrozenExpression]) \
            -> Iterator[Tuple[Substitution, Multiset[FrozenExpression]]]:
        """Match the multiset of expressions against the multiset of patterns using the bipartite graph.

        Because more expressions than patterns can be given, not all expression might be covered by a match.
        The remaining expressions are yielded along with the match substitution for every match.

        Args:
            expressions:
                A multiset of syntactic constant expressions. The cardinality of the expression multiset can be
                higher than that of the pattern multiset.
            patterns:
                A multiset of syntactic patterns.

        Yields:
            For every match, the corresponding substitution as well as the remaining expressions not covered by the
            match as a tuple.
        """
        subgraph = self._build_bipartite(expressions, patterns)
        matching_iter = enum_maximum_matchings_iter(subgraph)
        try:
            matching = next(matching_iter)
        except StopIteration:  # no matching found
            return
        if len(matching) < len(patterns):  # not all patterns covered by matching
            return

        for result, matched_expressions in self._substitution_from_matching(subgraph, matching):
            yield result, expressions - matched_expressions
        for matching in matching_iter:
            for result, matched_expressions in self._substitution_from_matching(subgraph, matching):
                yield result, expressions - matched_expressions

    def _substitution_from_matching(self, subgraph: Subgraph, matching: SubgraphMatching) \
            -> Iterator[Tuple[Substitution, Multiset[FrozenExpression]]]:
        """Create a match substitution from a bipartite matching.

        Args:
            subgraph:
                The bipartite subgraph used for matching.
            matching:
                The matching to create a substitution from if possible.

        Yields:
            If the matching is canonical and results in a valid substitution, that substitution is yielded along with
            the multiset of matched expressions.
        """
        # Limiting the matchings to canonical ones eliminates duplicate substitutions being yielded.
        if self._is_canonical_matching(matching):
            # The substitutions on each edge in the matching need to be unified
            # Only if they can be, it is a valid match
            substitutions = (subgraph[edge] for edge in matching.items())
            try:
                first_subst = next(substitutions)
                result = first_subst.union(*substitutions)
                matched_expressions = Multiset(subexpression for subexpression, _ in matching)
                yield result, matched_expressions
            except (ValueError, StopIteration):
                pass

    def _build_bipartite(self, expressions: Multiset[FrozenExpression], patterns: Multiset[FrozenExpression]) \
            -> Subgraph:
        """Construct the bipartite graph for the specific match situation.

        For the concrete matching, the bipartite graph must be limited to subexpressions and subpatterns actually
        occurring in the concrete expression and pattern. This is equivalent to taking the subgraph induced by the
        occurring subexpression and subpattern nodes.

        However, since the same subexpression and subpatterns can occur multiple times (hence the multisets), their
        corresponding nodes have to be multiplied by the number of occurrences. These nodes are labeled with a number
        as well as the original expression, resulting in a tuple.

        Args:
            expressions:
                The multiset of subexpressions of the concrete commutative expression.
            patterns:
                The multiset of subpatterns of the concrete commutative pattern.

        Returns:
            The multiplied induced bipartite subgraph for the concrete match situation.
        """
        bipartite = BipartiteGraph()  # type: Subgraph
        for (expr, patt), m in self.bipartite.edges_with_value():
            for i in range(expressions[expr]):
                for j in range(patterns[patt]):
                    bipartite[(expr, i), (patt, j)] = m
        return bipartite

    @staticmethod
    def _is_canonical_matching(matching: SubgraphMatching) -> bool:
        r"""Check if a matching is canonical.

        In the bipartite graph build by :meth:`_build_bipartite`, nodes for expressions and patterns are labeled with
        a number in addition to the expression or pattern itself. This is to handle multiple occurrences correctly.
        Consider this example::

            Expressions:    (g(a), 0)   (g(a), 1)
                                |    \  /    |
                                |     \/     |
                                |     /\     |
                                |    /  \    |
            Patterns:       (g(x_), 0)  (g(x_), 1)

        Here, there are two matchings in the bipartite graph, but both are equivalent. One matching uses the
        diagonal edges, the other the straight ones. However, if we define a unique canonical matching then we can avoid
        such duplications.

        A canonical matching has only edges where the number on the expression is lower than or equal to the number on
        the pattern. In the above example that applies to the matching with the straight edges only.

        Args:
            matching: The matching to check.

        Returns:
            ``True``, iff the matching is canonical.
        """
        for (_, i), (_, j) in matching.items():
            if i > j:
                return False
        return True
