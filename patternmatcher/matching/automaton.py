# -*- coding: utf-8 -*-
from collections import deque
from typing import Dict, Iterable, NamedTuple, Optional, Set, Union

from graphviz import Digraph
from multiset import Multiset

from ..constraints import Constraint, MultiConstraint
from ..expressions import (Expression, FrozenExpression, Operation,
                           Substitution, Symbol, SymbolWildcard, Variable,
                           Wildcard, freeze)
from ..utils import (VariableWithCount,
                     commutative_sequence_variable_partition_iter)
from .bipartite import BipartiteGraph, enum_maximum_matchings_iter
from .syntactic import OPERATION_END, is_operation

__all__ = ['Automaton']

LabelType = Union[FrozenExpression, type]
_State = NamedTuple('_State', [
    ('transitions', Dict[LabelType, '_Transition']),
    ('patterns', Set[int]),
    ('matcher', Optional['CommutativeMatcher'])
])

_Transition = NamedTuple('_Transition', [
    ('label', LabelType),
    ('target', _State),
    ('constraint', Optional[Constraint]),
    ('variable_name', Optional[str])
])


class Automaton:
    __slots__ = ('patterns', 'states', 'root', 'pattern_vars')

    def __init__(self):
        self.patterns = []
        self.states = []
        self.root = self._create_state()
        self.pattern_vars = []

    def add(self, pattern):
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
        expressions_stack = [deque([pattern])]
        context_stack = []

        while expressions_stack:
            if expressions_stack[-1]:
                expr = expressions_stack[-1].popleft()
                variable_name = None
                constraint = None
                if isinstance(expr, Variable):
                    constraint = expr.constraint
                    variable_name = expr.name
                    expr = expr.expression
                if isinstance(expr, Operation):
                    if expr.commutative:
                        commutative = True
                    else:
                        context_stack.append(MultiConstraint.create(constraint, expr.constraint))
                        expressions_stack.append(deque(expr.operands))
                        constraint = None
                constraint = MultiConstraint.create(constraint, expr.constraint)
                state = self._create_expression_transition(state, expr, constraint, variable_name, pattern_index)
                index += 1
            else:
                expressions_stack.pop()
                if context_stack:
                    constraint = context_stack.pop()
                    state = self._create_simple_transition(state, OPERATION_END, constraint)

        state.patterns.add(pattern_index)

        return pattern_index

    def match(self, expression):
        expression = freeze(expression)

        for state, substitution in self._match(self.root, [expression], Substitution()):
            for pattern_index in state.patterns:
                renaming = self.pattern_vars[pattern_index]
                substitution = substitution.rename({renamed: original for original, renamed in renaming.items()})
                yield self.patterns[pattern_index], substitution

    def as_graph(self, finals=None) -> Digraph:  # pragma: no cover
        dot = Digraph()

        for state in self.states:
            name = 'n{!s}'.format(id(state))
            if finals is not None and not state.transitions:
                finals.append(name)
            if state.matcher:
                dot.node(name, 'Sub Matcher', {'shape': 'box'})
                subfinals = []
                dot.subgraph(state.matcher.automaton.as_graph(subfinals))
                submatch_label = 'Sub Matcher End'
                for pattern_index, subpatterns, variables in state.matcher.patterns.values():
                    var_formatted = ', '.join('{}[{}]x{}'.format(n, m, c) for n, c, m in variables)
                    submatch_label += '\n{}: {} {}'.format(pattern_index, subpatterns, var_formatted)
                dot.node(name+'-end', submatch_label, {'shape': 'box'})
                for f in subfinals:
                    dot.edge(f, name+'-end')
                dot.edge(name, 'n{}'.format(id(state.matcher.automaton.root)))
            elif not state.patterns:
                dot.node(name, '', {'shape': ('circle' if state.transitions else 'doublecircle')})
            else:
                vars = ['{}: {}'.format(p, repr(self.pattern_vars[p])) for p in state.patterns]
                label = '\n'.join(vars)
                dot.node(name, label, {'shape': 'box'})

        for state in self.states:
            for head, transitions in state.transitions.items():
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
                    dot.edge(start, end, t_label)

        return dot

    def _create_expression_transition(self, state, expr, constraint, variable_name, pattern_index):
        label, head = self._get_label_and_head(expr)
        transitions = state.transitions.setdefault(head, [])
        commutative = getattr(expr, 'commutative', False)
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
                matcher = CommutativeMatcher(expr.associative)
            state = self._create_state(matcher)
            transition = _Transition(label, state, constraint, variable_name)
            transitions.append(transition)
        if commutative:
            subpattern_id = matcher.add_pattern(pattern_index, expr.operands)
            return self._create_simple_transition(state, subpattern_id, constraint)
        return state

    def _create_simple_transition(self, state, label, constraint):
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

    def _get_label_and_head(self, expr):
        if isinstance(expr, Operation):
            label = type(expr)
            head = label._original_base
        else:
            label = expr.without_constraints
            head = expr.head
            if isinstance(label, SymbolWildcard):
                head = label.symbol_type
        return label, head

    def _match(self, state, expressions, substitution, associative=None):
        if not state.transitions:
            if state.patterns and not expressions:
                yield state, substitution
            return

        if len(expressions) == 0 and OPERATION_END in state.transitions:
            yield state, substitution

        expression = expressions[0] if expressions else None
        heads = [expression.head] if expression else []
        if isinstance(expression, Symbol):
            heads.extend(base for base in type(expression).__mro__ if issubclass(base, Symbol) and not issubclass(base, FrozenExpression))
        heads.append(None)

        for head in heads:
            if head in state.transitions:
                for transition in state.transitions[head]:
                    label = transition.label
                    matched_expr = None
                    new_expressions = None
                    if is_operation(label):
                        if isinstance(expression, label):
                            matched_expr = expression
                            new_expressions = expressions[1:]
                            matcher = transition.target.matcher
                            if matcher:
                                for operand in expression.operands:
                                    matcher.add_subject(operand)
                                for matched_pattern, new_subst in matcher.match(expression.operands, substitution):
                                    transition_set = transition.target.transitions[matched_pattern]
                                    for next_transition in transition_set:
                                        eventual_subst = self._check_constraint(next_transition, new_subst, matched_expr)
                                        if eventual_subst is not None:
                                            yield from self._match(next_transition.target, new_expressions, eventual_subst, associative)
                            else:
                                for new_state, new_subst in self._match(transition.target, expression.operands, substitution, label if label.associative else None):
                                    if OPERATION_END in new_state.transitions:
                                        for transition in new_state.transitions[OPERATION_END]:
                                            eventual_subst = self._check_constraint(transition, new_subst, matched_expr)
                                            if eventual_subst is not None:
                                                yield from self._match(transition.target, new_expressions, eventual_subst, associative)
                            continue
                    elif isinstance(label, Symbol):
                        if label == expression:
                            matched_expr = expression
                            new_expressions = expressions[1:]
                    elif isinstance(label, SymbolWildcard):
                        if isinstance(expression, label.symbol_type):
                            matched_expr = expression
                            new_expressions = expressions[1:]
                    elif isinstance(label, Wildcard):
                        min_count = label.min_count
                        if label.fixed_size and not associative:
                            if min_count == 1:
                                if expression is not None:
                                    matched_expr = expression
                                    new_expressions = expressions[1:]
                            elif len(expressions) >= min_count:
                                matched_expr = tuple(expressions[:min_count])
                                new_expressions = expressions[min_count:]
                        else:
                            for i in range(min_count, len(expressions) + 1):
                                matched_expr = tuple(expressions[:i])
                                if associative and label.fixed_size:
                                    if i > min_count:
                                        wrapped = associative.from_args(*matched_expr[min_count-1:])
                                        if min_count == 1:
                                            matched_expr = wrapped
                                        else:
                                            matched_expr = matched_expr[:min_count-1] + (wrapped, )
                                    elif min_count == 1:
                                        matched_expr = matched_expr[0]
                                new_expressions = expressions[i:]
                                new_subst = self._check_constraint(transition, substitution, matched_expr)
                                if new_subst is not None:
                                    yield from self._match(transition.target, new_expressions, new_subst, associative)
                            continue

                    if new_expressions is not None:
                        new_subst = self._check_constraint(transition, substitution, matched_expr)
                        if new_subst is not None:
                            yield from self._match(transition.target, new_expressions, new_subst, associative)

    def _create_state(self, matcher=None):
        state = _State(dict(), set(), matcher)
        self.states.append(state)
        return state

    @staticmethod
    def _is_sequence_wildcard(wc, associative=False):
        if isinstance(wc, Variable):
            wc = wc.expression
        if isinstance(wc, SymbolWildcard):
            return False
        if isinstance(wc, Wildcard):
            return not wc.fixed_size or associative
        return False

    @classmethod
    def _collect_variable_renaming(cls, expr, position=None, variables=None):
        if position is None:
            position = [0]
        if variables is None:
            variables = {}
        if isinstance(expr, Variable):
            if expr.name not in variables:
                new_name = 'i{}'.format('.'.join(map(str, position)))
                if new_name in variables.values():
                    counter = 1
                    while '{}_{}'.format(new_name, counter) in variables.values():
                        counter += 1
                    new_name = '{}_{}'.format(new_name, counter)
                variables[expr.name] = new_name
            expr = expr.expression
        position[-1] += 1
        if isinstance(expr, Operation):
            if expr.commutative:
                for operand in expr.operands:
                    cls._collect_variable_renaming(operand, position + [0], variables)
            else:
                for operand in expr.operands:
                    cls._collect_variable_renaming(operand, position, variables)

        return variables

    @staticmethod
    def _check_constraint(transition, substitution, expr):
        if transition.variable_name is not None:
            try:
                substitution = substitution.union_with_variable(transition.variable_name, expr)
            except ValueError:
                return None
        if transition.constraint is None:
            return substitution
        return substitution if transition.constraint(substitution) else None


class CommutativeMatcher(object):
    __slots__ = ('patterns', 'subjects', 'automaton', 'bipartite', 'associative')

    def __init__(self, associative):
        self.patterns = {}
        self.subjects = {}
        self.automaton = Automaton()
        self.bipartite = BipartiteGraph()
        self.associative = associative

    def add_pattern(self, pattern_index: int, operands: Iterable[FrozenExpression]) -> None:
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
            subject_id = self.subjects[subject] = len(self.subjects)
            self.subjects[subject_id] = subject
            for state, parts in self.automaton._match(self.automaton.root, [subject], Substitution(), self.associative):
                for pattern_index in state.patterns:
                    variables = self.automaton.pattern_vars[pattern_index]
                    substitution = Substitution((name, parts[index]) for name, index in variables.items())
                    self.bipartite[subject_id, pattern_index] = substitution
        else:
            subject_id = self.subjects[subject]
        return subject_id

    def match(self, subjects, substitution):
        subject_ids = Multiset(self.subjects[s] for s in subjects)
        for pattern_index, pattern_set, pattern_vars in self.patterns.values():
            if pattern_set:
                bipartite_match_iter = self._match_with_bipartite(subject_ids, pattern_set, substitution)
                for bipartite_subst, matched_subjects in bipartite_match_iter:
                    if pattern_vars:
                        remaining = Multiset(self.subjects[s] for s in (subject_ids - matched_subjects))
                        sequence_var_iter = self._match_sequence_variables(remaining, pattern_vars, bipartite_subst)
                        for result_substitution in sequence_var_iter:
                            yield pattern_index, result_substitution
                    elif len(subjects) == len(pattern_set):
                        yield pattern_index, bipartite_subst
            elif pattern_vars:
                sequence_var_iter = self._match_sequence_variables(Multiset(subjects), pattern_vars, substitution)
                for variable_substitution in sequence_var_iter:
                    yield pattern_index, variable_substitution
            elif len(subjects) == 0:
                yield pattern_index, substitution

    def _extract_sequence_wildcards(self, operands):
        pattern_set = Multiset()
        pattern_vars = dict()
        for operand in operands:
            if not self._is_sequence_wildcard(operand):
                pattern_set.add(self.automaton.add(operand))
            else:
                varname = getattr(operand, 'name', None)
                wc = operand.expression if isinstance(operand, Variable) else operand
                if isinstance(wc, Wildcard):
                    if varname in pattern_vars:
                        _, count, _ = pattern_vars[varname]
                    else:
                        count = 0
                    pattern_vars[varname] = VariableWithCount(varname, count + 1, wc.min_count)
        return pattern_set, pattern_vars

    def _is_sequence_wildcard(self, wc):
        if isinstance(wc, Variable):
            wc = wc.expression
        if isinstance(wc, SymbolWildcard):
            return False
        if isinstance(wc, Wildcard):
            return not wc.fixed_size or self.associative
        return False

    def _match_with_bipartite(self, subject_ids, pattern_set, substitution):
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
            print(bipartite_substitution, matched_subjects)
            yield bipartite_substitution, matched_subjects

    @staticmethod
    def _match_sequence_variables(subjects, pattern_vars, substitution):
        for variable_substitution in commutative_sequence_variable_partition_iter(subjects, pattern_vars):
            try:
                yield substitution.union(variable_substitution)
            except ValueError:
                pass

    def _build_bipartite(self, subjects, patterns):
        bipartite = BipartiteGraph()
        for (expr, patt), m in self.bipartite.items():
            for i in range(subjects[expr]):
                for j in range(patterns[patt]):
                    bipartite[(expr, i), (patt, j)] = m
        return bipartite

    @staticmethod
    def _is_canonical_matching(matching, subject_ids, pattern_set):
        inverted_matching = {p: s for s, p in matching.items()}
        for (pattern_index, count) in sorted(pattern_set.items()):
            _, previous_label = inverted_matching[pattern_index, 0]
            for i in range(1, count):
                _, label = inverted_matching[pattern_index, i]
                if label < previous_label:
                    return False
                previous_label = label
        for subject_id, count in subject_ids.items():
            patterns = iter(matching[subject_id, i] for i in range(count) if (subject_id, i) in matching)
            try:
                last_pattern = next(patterns)
            except StopIteration:
                pass
            else:
                for next_pattern in patterns:
                    if next_pattern < last_pattern:
                        return False
                    last_pattern = next_pattern
        return True
