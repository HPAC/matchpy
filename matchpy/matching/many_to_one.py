# -*- coding: utf-8 -*-
"""Contains the :class:`ManyToOneMatcher` which can be used for fast many-to-one matching.

You can initialize the matcher with a list of the patterns that you wish to match:

>>> pattern1 = f(a, x_)
>>> pattern2 = f(y_, b)
>>> matcher = ManyToOneMatcher(pattern1, pattern2)

You can also add patterns later:

>>> pattern3 = f(a, b)
>>> _ = matcher.add(pattern3)

Then you can match a subject against all the patterns at once:

>>> subject = f(a, b)
>>> for matched_pattern, substitution in sorted(matcher.match(subject)):
...     print('{} matched with {}'.format(matched_pattern, substitution))
f(a, b) matched with {}
f(a, x_) matched with {x ↦ b}
f(y_, b) matched with {y ↦ a}
"""

import math
from collections import deque
from typing import (Container, Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple, Type, Union)

from graphviz import Digraph
from multiset import Multiset

from ..expressions import (
    Constraint, Expression, FrozenExpression, MultiConstraint, Operation, Substitution, Symbol, SymbolWildcard,
    Variable, Wildcard, freeze
)
from ..utils import (VariableWithCount, commutative_sequence_variable_partition_iter)
from .bipartite import BipartiteGraph, enum_maximum_matchings_iter
from .syntactic import OPERATION_END, is_operation

__all__ = ['ManyToOneMatcher']

LabelType = Union[FrozenExpression, Type[Operation]]
HeadType = Optional[Union[FrozenExpression, Type[Operation], Type[Symbol]]]

_State = NamedTuple('_State', [
    ('transitions', Dict[LabelType, '_Transition']),
    ('patterns', Set[int]),
    ('matcher', Optional['CommutativeMatcher'])
])  # yapf: disable

_Transition = NamedTuple('_Transition', [
    ('label', LabelType),
    ('target', _State),
    ('constraint', Optional[Constraint]),
    ('variable_name', Optional[str])
])  # yapf: disable

_MatchContext = NamedTuple('_MatchContext', [
    ('subjects', Tuple[FrozenExpression]),
    ('substitution', Substitution),
    ('associative', Optional[type])
])  # yapf: disable


class ManyToOneMatcher:
    __slots__ = ('patterns', 'states', 'root', 'pattern_vars')

    def __init__(self, *patterns: Expression) -> None:
        """
        Args:
            *patterns: The patterns which the matcher should match.
        """
        self.patterns = []
        self.states = []
        self.root = self._create_state()
        self.pattern_vars = []

        for pattern in patterns:
            self.add(pattern)

    def add(self, pattern: Expression) -> int:
        """Add a new pattern to the matcher.

        Equivalent patterns are not added again. However, patterns that are structurally equivalent,
        but have different constraints or different variable names are distinguished by the matcher.

        Args:
            pattern: The pattern to add.

        Returns:
            The internal id for the pattern. This is mainly used by the :class:`CommutativeMatcher`.
        """
        try:
            return self.patterns.index(pattern)
        except ValueError:
            pass

        pattern_index = len(self.patterns)
        renaming = self._collect_variable_renaming(pattern)
        index = 0
        self.patterns.append(pattern)
        self.pattern_vars.append(renaming)
        pattern = freeze(pattern.with_renamed_vars(renaming))
        state = self.root
        patterns_stack = [deque([pattern])]
        context_stack = []

        while patterns_stack:
            if patterns_stack[-1]:
                subpattern = patterns_stack[-1].popleft()
                variable_name = None
                constraint = None
                has_pre_constraint = True
                if isinstance(subpattern, Variable):
                    constraint = subpattern.constraint
                    variable_name = subpattern.name
                    subpattern = subpattern.expression
                constraint = MultiConstraint.create(constraint, subpattern.constraint)
                if isinstance(subpattern, Operation):
                    if not subpattern.commutative:
                        context_stack.append(constraint)
                        patterns_stack.append(deque(subpattern.operands))
                    has_pre_constraint = False
                state = self._create_expression_transition(
                    state, subpattern, constraint if has_pre_constraint else None, variable_name
                )
                if getattr(subpattern, 'commutative', False):
                    subpattern_id = state.matcher.add_pattern(subpattern.operands)
                    state = self._create_simple_transition(state, subpattern_id, constraint)
                index += 1
            else:
                patterns_stack.pop()
                if context_stack:
                    constraint = context_stack.pop()
                    state = self._create_simple_transition(state, OPERATION_END, constraint)

        state.patterns.add(pattern_index)

        return pattern_index

    def match(self, subject: Expression) -> Iterator[Tuple[Expression, Substitution]]:
        """Match the subject against all the matcher's patterns.

        Args:
            subject: The subject to match.

        Yields:
            For every match, a tuple of the matching pattern and the match substitution.
        """
        subject = freeze(subject)
        context = _MatchContext((subject, ), Substitution(), None)
        for state, substitution in self._match(self.root, context):
            for pattern_index in state.patterns:
                renaming = self.pattern_vars[pattern_index]
                new_substitution = substitution.rename({renamed: original for original, renamed in renaming.items()})
                yield self.patterns[pattern_index], new_substitution

    def as_graph(self) -> Digraph:  # pragma: no cover
        return self._as_graph(None)

    def _as_graph(self, finals: Optional[List[str]]) -> Digraph:  # pragma: no cover
        graph = Digraph()
        self._make_graph_nodes(graph, finals)
        self._make_graph_edges(graph)
        return graph

    def _make_graph_nodes(self, graph: Digraph, finals: Optional[List[str]]) -> None:  # pragma: no cover
        for state in self.states:
            name = 'n{!s}'.format(id(state))
            if finals is not None and not state.transitions:
                finals.append(name)
            if state.matcher:
                graph.node(name, 'Sub Matcher', {'shape': 'box'})
                subfinals = []
                graph.subgraph(state.matcher.automaton._as_graph(subfinals))
                submatch_label = 'Sub Matcher End'
                for pattern_index, subpatterns, variables in state.matcher.patterns.values():
                    var_formatted = ', '.join('{}[{}]x{}'.format(n, m, c) for n, c, m in variables)
                    submatch_label += '\n{}: {} {}'.format(pattern_index, subpatterns, var_formatted)
                graph.node(name + '-end', submatch_label, {'shape': 'box'})
                for f in subfinals:
                    graph.edge(f, name + '-end')
                graph.edge(name, 'n{}'.format(id(state.matcher.automaton.root)))
            elif not state.patterns:
                graph.node(name, '', {'shape': ('circle' if state.transitions else 'doublecircle')})
            else:
                variables = ['{}: {}'.format(p, repr(self.pattern_vars[p])) for p in state.patterns]
                label = '\n'.join(variables)
                graph.node(name, label, {'shape': 'box'})

    def _make_graph_edges(self, graph: Digraph) -> None:  # pragma: no cover
        for state in self.states:
            for _, transitions in state.transitions.items():
                for transition in transitions:
                    t_label = str(transition.label)
                    if is_operation(transition.label):
                        t_label += '('
                    if transition.constraint is not None:
                        t_label += ' ;/ ' + str(transition.constraint)
                    if transition.variable_name:
                        t_label += '\n \\=> {}'.format(transition.variable_name)

                    start = 'n{!s}'.format(id(state))
                    if state.matcher:
                        start += '-end'
                    end = 'n{!s}'.format(id(transition.target))
                    graph.edge(start, end, t_label)

    def _create_expression_transition(
            self, state: _State, expression: Expression, constraint: Optional[Constraint], variable_name: Optional[str]
    ) -> _State:
        label, head = self._get_label_and_head(expression)
        transitions = state.transitions.setdefault(head, [])
        commutative = getattr(expression, 'commutative', False)
        pre_constraint = constraint if not commutative else None
        matcher = None
        for transition in transitions:
            if (transition.constraint == pre_constraint and
                    transition.variable_name == variable_name and
                    transition.label == label):
                state = transition.target
                matcher = state.matcher
                break
        else:
            if commutative:
                matcher = CommutativeMatcher(expression.associative)
            state = self._create_state(matcher)
            transition = _Transition(label, state, constraint, variable_name)
            transitions.append(transition)
        return state

    def _create_simple_transition(self, state: _State, label: LabelType, constraint: Optional[Constraint]) -> _State:
        transitions = state.transitions.setdefault(label, [])
        for transition in transitions:
            if transition.constraint == constraint:
                state = transition.target
                break
        else:
            state = self._create_state()
            transition = _Transition(label, state, constraint, None)
            transitions.append(transition)
        return state

    @staticmethod
    def _get_label_and_head(expression: Expression) -> Tuple[LabelType, HeadType]:
        if isinstance(expression, Operation):
            label = type(expression)
            head = label._original_base
        else:
            label = expression.without_constraints
            head = expression.head
            if isinstance(label, SymbolWildcard):
                head = label.symbol_type
        return label, head

    def _match(self, state: _State, context: _MatchContext) -> Iterator[Tuple[_State, Substitution]]:
        subjects, substitution, _ = context
        if not state.transitions:
            if state.patterns and not subjects:
                yield state, substitution
            return

        if len(subjects) == 0 and OPERATION_END in state.transitions:
            yield state, substitution

        subject = subjects[0] if subjects else None
        heads = self._get_heads(subject)

        for head in heads:
            for transition in state.transitions.get(head, []):
                yield from self._match_transition(transition, context)

    def _match_transition(self, transition: _Transition,
                          context: _MatchContext) -> Iterator[Tuple[_State, Substitution]]:
        subjects, substitution, associative = context
        label = transition.label
        subject = subjects[0] if subjects else None
        matched_subject = subject
        new_subjects = subjects[1:]
        if is_operation(label):
            if transition.target.matcher:
                yield from self._match_commutative_operation(transition.target, context)
            else:
                yield from self._match_regular_operation(transition, context)
            return
        elif isinstance(label, Wildcard) and not isinstance(label, SymbolWildcard):
            min_count = label.min_count
            if label.fixed_size and not associative:
                if min_count == 1:
                    if subject is None:
                        return
                elif len(subjects) >= min_count:
                    matched_subject = tuple(subjects[:min_count])
                    new_subjects = subjects[min_count:]
                else:
                    return
            else:
                yield from self._match_sequence_variable(label, transition, context)
                return

        new_substitution = self._check_constraint(transition, substitution, matched_subject)
        if new_substitution is not None:
            new_context = _MatchContext(new_subjects, new_substitution, associative)
            yield from self._match(transition.target, new_context)

    @staticmethod
    def _get_heads(expression: Expression) -> List[HeadType]:
        heads = [expression.head] if expression else []
        if isinstance(expression, Symbol):
            heads.extend(
                base for base in type(expression).__mro__
                if issubclass(base, Symbol) and not issubclass(base, FrozenExpression)
            )
        heads.append(None)
        return heads

    def _match_sequence_variable(self, wildcard: Wildcard, transition: _Transition,
                                 context: _MatchContext) -> Iterator[Tuple[_State, Substitution]]:
        subjects, substitution, associative = context
        min_count = wildcard.min_count
        for i in range(min_count, len(subjects) + 1):
            matched_subject = tuple(subjects[:i])
            if associative and wildcard.fixed_size:
                if i > min_count:
                    wrapped = associative.from_args(*matched_subject[min_count - 1:])
                    if min_count == 1:
                        matched_subject = wrapped
                    else:
                        matched_subject = matched_subject[:min_count - 1] + (wrapped, )
                elif min_count == 1:
                    matched_subject = matched_subject[0]
            new_substitution = self._check_constraint(transition, substitution, matched_subject)
            if new_substitution is not None:
                new_context = _MatchContext(subjects[i:], new_substitution, associative)
                yield from self._match(transition.target, new_context)

    def _match_commutative_operation(self, state: _State,
                                     context: _MatchContext) -> Iterator[Tuple[_State, Substitution]]:
        (subject, *rest), substitution, associative = context
        matcher = state.matcher
        for operand in subject.operands:
            matcher.add_subject(operand)
        for matched_pattern, new_substitution in matcher.match(subject.operands, substitution):
            transition_set = state.transitions[matched_pattern]
            for next_transition in transition_set:
                eventual_substitution = self._check_constraint(next_transition, new_substitution, subject)
                if eventual_substitution is not None:
                    new_context = _MatchContext(rest, eventual_substitution, associative)
                    yield from self._match(next_transition.target, new_context)

    def _match_regular_operation(self, transition: _Transition,
                                 context: _MatchContext) -> Iterator[Tuple[_State, Substitution]]:
        (subject, *rest), substitution, associative = context
        new_associative = transition.label if transition.label.associative else None
        new_context = _MatchContext(subject.operands, substitution, new_associative)
        for new_state, new_substitution in self._match(transition.target, new_context):
            for transition in new_state.transitions[OPERATION_END]:
                eventual_substitution = self._check_constraint(transition, new_substitution, subject)
                if eventual_substitution is not None:
                    new_context = _MatchContext(rest, eventual_substitution, associative)
                    yield from self._match(transition.target, new_context)

    def _create_state(self, matcher: 'CommutativeMatcher' =None) -> _State:
        state = _State(dict(), set(), matcher)
        self.states.append(state)
        return state

    @classmethod
    def _collect_variable_renaming(
            cls, expression: Expression, position: List[int]=None, variables: Dict[str, str]=None
    ) -> Dict[str, str]:
        """Return renaming for the variables in the expression.

        The variable names are generated according to the position of the variable in the expression. The goal is to
        rename variables in structurally identical patterns so that the automaton contains less redundant states.
        """
        if position is None:
            position = [0]
        if variables is None:
            variables = {}
        if isinstance(expression, Variable):
            if expression.name not in variables:
                variables[expression.name] = cls._get_name_for_position(position, variables.values())
            expression = expression.expression
        position[-1] += 1
        if isinstance(expression, Operation):
            if expression.commutative:
                for operand in expression.operands:
                    position.append(0)
                    cls._collect_variable_renaming(operand, position, variables)
                    position.pop()
            else:
                for operand in expression.operands:
                    cls._collect_variable_renaming(operand, position, variables)

        return variables

    @staticmethod
    def _get_name_for_position(position: List[int], variables: Container[str]) -> str:
        new_name = 'i{}'.format('.'.join(map(str, position)))
        if new_name in variables:
            counter = 1
            while '{}_{}'.format(new_name, counter) in variables:
                counter += 1
            new_name = '{}_{}'.format(new_name, counter)
        return new_name

    @staticmethod
    def _check_constraint(transition: _Transition, substitution: Substitution,
                          expression: Expression) -> Optional[Substitution]:
        if transition.variable_name is not None:
            try:
                substitution = substitution.union_with_variable(transition.variable_name, expression)
            except ValueError:
                return None
        if transition.constraint is None:
            return substitution
        return substitution if transition.constraint(substitution) else None


Subgraph = BipartiteGraph[Tuple[int, int], Tuple[int, int], Substitution]
Matching = Dict[Tuple[int, int], Tuple[int, int]]


class CommutativeMatcher(object):
    __slots__ = ('patterns', 'subjects', 'automaton', 'bipartite', 'associative')

    def __init__(self, associative: Optional[type]) -> None:
        self.patterns = {}
        self.subjects = {}
        self.automaton = ManyToOneMatcher()
        self.bipartite = BipartiteGraph()
        self.associative = associative

    def add_pattern(self, operands: Iterable[FrozenExpression]) -> int:
        pattern_set, pattern_vars = self._extract_sequence_wildcards(operands)
        sorted_vars = tuple(sorted(pattern_vars.values()))
        sorted_subpatterns = tuple(sorted(pattern_set))
        pattern_key = sorted_subpatterns + sorted_vars
        if pattern_key not in self.patterns:
            inserted_id = len(self.patterns)
            self.patterns[pattern_key] = (inserted_id, pattern_set, sorted_vars)
        else:
            inserted_id = self.patterns[pattern_key][0]
        return inserted_id

    def add_subject(self, subject: FrozenExpression) -> None:
        if subject not in self.subjects:
            subject_id, pattern_set = self.subjects[subject] = (len(self.subjects), set())
            self.subjects[subject_id] = subject
            context = _MatchContext((subject, ), Substitution(), self.associative)
            for state, parts in self.automaton._match(self.automaton.root, context):
                for pattern_index in state.patterns:
                    variables = self.automaton.pattern_vars[pattern_index]
                    substitution = Substitution((name, parts[index]) for name, index in variables.items())
                    self.bipartite[subject_id, pattern_index] = substitution
                    pattern_set.add(pattern_index)
        else:
            subject_id, _ = self.subjects[subject]
        return subject_id

    def match(self, subjects: Sequence[FrozenExpression],
              substitution: Substitution) -> Iterator[Tuple[int, Substitution]]:
        subject_ids = Multiset()
        pattern_ids = Multiset()
        for subject in subjects:
            subject_id, subject_pattern_ids = self.subjects[subject]
            subject_ids.add(subject_id)
            pattern_ids.update(subject_pattern_ids)
        for pattern_index, pattern_set, pattern_vars in self.patterns.values():
            if pattern_set:
                if not pattern_set <= pattern_ids:
                    continue
                bipartite_match_iter = self._match_with_bipartite(subject_ids, pattern_set, substitution)
                for bipartite_substitution, matched_subjects in bipartite_match_iter:
                    if pattern_vars:
                        remaining_ids = subject_ids - matched_subjects
                        remaining = Multiset(self.subjects[id] for id in remaining_ids)  # pylint: disable=not-an-iterable
                        sequence_var_iter = self._match_sequence_variables(
                            remaining, pattern_vars, bipartite_substitution
                        )
                        for result_substitution in sequence_var_iter:
                            yield pattern_index, result_substitution
                    elif len(subjects) == len(pattern_set):
                        yield pattern_index, bipartite_substitution
            elif pattern_vars:
                sequence_var_iter = self._match_sequence_variables(Multiset(subjects), pattern_vars, substitution)
                for variable_substitution in sequence_var_iter:
                    yield pattern_index, variable_substitution
            elif len(subjects) == 0:
                yield pattern_index, substitution

    def _extract_sequence_wildcards(self, operands: Iterable[Expression]
                                   ) -> Tuple[Multiset[int], Dict[str, VariableWithCount]]:
        pattern_set = Multiset()
        pattern_vars = dict()
        for operand in operands:
            if not self._is_sequence_wildcard(operand):
                pattern_set.add(self.automaton.add(operand))
            else:
                varname = getattr(operand, 'name', None)
                wildcard = operand.expression if isinstance(operand, Variable) else operand
                if varname is None:
                    if varname in pattern_vars:
                        _, _, min_count = pattern_vars[varname]
                    else:
                        min_count = 0
                    pattern_vars[varname] = VariableWithCount(varname, 1, wildcard.min_count + min_count)
                else:
                    if varname in pattern_vars:
                        _, count, _ = pattern_vars[varname]
                    else:
                        count = 0
                    pattern_vars[varname] = VariableWithCount(varname, count + 1, wildcard.min_count)
        return pattern_set, pattern_vars

    def _is_sequence_wildcard(self, expression: Expression) -> bool:
        if isinstance(expression, Variable):
            expression = expression.expression
        if isinstance(expression, SymbolWildcard):
            return False
        if isinstance(expression, Wildcard):
            return not expression.fixed_size or self.associative
        return False

    def _match_with_bipartite(
            self,
            subject_ids: Multiset[int],
            pattern_set: Multiset[int],
            substitution: Substitution,
    ) -> Iterator[Tuple[Substitution, Multiset[int]]]:
        bipartite = self._build_bipartite(subject_ids, pattern_set)
        for matching in enum_maximum_matchings_iter(bipartite):
            if len(matching) < len(pattern_set):
                break
            if not self._is_canonical_matching(matching, subject_ids, pattern_set):
                continue
            try:
                bipartite_substitution = substitution.union(*(bipartite[edge] for edge in matching.items()))
            except ValueError:
                continue
            matched_subjects = Multiset(subexpression for subexpression, _ in matching)
            yield bipartite_substitution, matched_subjects

    @staticmethod
    def _match_sequence_variables(
            subjects: Multiset[FrozenExpression],
            pattern_vars: Sequence[VariableWithCount],
            substitution: Substitution,
    ) -> Iterator[Substitution]:
        for variable_substitution in commutative_sequence_variable_partition_iter(subjects, pattern_vars):
            try:
                print(variable_substitution)
                yield substitution.union(variable_substitution)
            except ValueError:
                pass

    def _build_bipartite(self, subjects: Multiset[int], patterns: Multiset[int]) -> Subgraph:
        bipartite = BipartiteGraph()
        for (expression, pattern), substitution in self.bipartite.edges_with_value():
            for i in range(subjects[expression]):
                for j in range(patterns[pattern]):
                    bipartite[(expression, i), (pattern, j)] = substitution
        return bipartite

    @staticmethod
    def _is_canonical_matching(matching: Matching, subject_ids: Multiset[int], pattern_set: Multiset[int]) -> bool:
        inverted_matching = {p: s for s, p in matching.items()}
        for (pattern_index, count) in sorted(pattern_set.items()):
            _, previous_label = inverted_matching[pattern_index, 0]
            for i in range(1, count):
                _, label = inverted_matching[pattern_index, i]
                if label < previous_label:
                    return False
                previous_label = label
        for subject_id, count in subject_ids.items():
            patterns = iter(matching.get((subject_id, i), (math.inf, math.inf)) for i in range(count))
            last_pattern = next(patterns)
            for next_pattern in patterns:
                if next_pattern < last_pattern:
                    return False
                last_pattern = next_pattern
        return True
