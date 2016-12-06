# -*- coding: utf-8 -*-
from collections import deque
from typing import NamedTuple, Union, Dict, Set, Optional, Iterable
import itertools
import operator

from graphviz import Digraph
from multiset import Multiset

from ..expressions import Operation, Variable, Wildcard, Symbol, SymbolWildcard, freeze, Substitution, FrozenExpression, Expression
from .syntactic import OPERATION_END, is_operation, is_symbol_wildcard, FlatTerm
from ..constraints import MultiConstraint, EqualVariablesConstraint, Constraint
from .common import CommutativePatternsParts, match
from .bipartite import BipartiteGraph, enum_maximum_matchings_iter
from ..utils import VariableWithCount, commutative_sequence_variable_partition_iter

__all__ = ['Automaton']

def _term_str(term) -> str:  # pragma: no cover
    """Return a string representation of a term atom."""
    if is_operation(term):
        return term.name + '('
    #elif isinstance(term, SymbolWildcard):
    #    return '*{!s}'.format(term.__name__)
    #elif isinstance(term, Wildcard):
    #    return '*{!s}{!s}'.format(term.min_count, (not term.fixed_size) and '+' or '')
    else:
        return str(term)

LabelType = Union[Expression,type]
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
    def __init__(self):
        self.patterns = []
        self.states = []
        self.root = self._create_state()
        self.pattern_vars = []
        self.matchers = []
        self.var_prefix = ''

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

    def add(self, pattern):
        try:
            return self.patterns.index(pattern)
        except ValueError:
            pass

        pattern_id = len(self.patterns)
        renaming = self._collect_variable_renaming(pattern)
        index = 0
        self.patterns.append(pattern)
        self.pattern_vars.append(renaming)
        pattern = freeze(pattern.with_renamed_vars(renaming))
        state = self.root
        expressions_stack = [deque([pattern])]
        context_stack = []
        subpattern_id = None

        while expressions_stack:
            if expressions_stack[-1]:
                expr = expressions_stack[-1].popleft()
                label = None
                variable_name = None
                constraint = None
                commutative = False
                if isinstance(expr, Variable):
                    constraint = expr.constraint
                    variable_name = expr.name
                    expr = expr.expression
                if isinstance(expr, Operation):
                    label = type(expr)
                    head = label._original_base
                    context_stack.append((MultiConstraint.create(constraint, expr.constraint), variable_name))
                    if label.commutative:
                        commutative = True
                        expressions_stack.append(deque())
                    else:
                        expressions_stack.append(deque(expr.operands))
                else:
                    label = expr.without_constraints
                    head = label.head
                    if isinstance(label, SymbolWildcard):
                        head = label.symbol_type
                constraint = MultiConstraint.create(constraint, expr.constraint)
                if label is not None:
                    transitions = state.transitions.setdefault(head, [])
                    matcher = None
                    for transition in transitions:
                        if transition.constraint == constraint and transition.variable_name == variable_name:
                            state = transition.target
                            matcher = state.matcher
                            break
                    else:
                        if commutative:
                            matcher = CommutativeMatcher(expr.associative)
                            self.matchers.append(matcher)
                        state = self._create_state(matcher)
                        transition = _Transition(label, state, constraint, variable_name)
                        transitions.append(transition)
                    if commutative:
                        subpattern_id = matcher.add_pattern(pattern_id, expr.operands)
                index += 1
            else:
                expressions_stack.pop()
                if context_stack:
                    label = subpattern_id if subpattern_id is not None else OPERATION_END
                    subpattern_id = None
                    constraint, variable_name = context_stack.pop()
                    transitions = state.transitions.setdefault(label, [])
                    for transition in transitions:
                        if transition.constraint == constraint and transition.variable_name == variable_name:
                            state = transition.target
                            break
                    else:
                        state = self._create_state()
                        transition = _Transition(label, state, constraint, variable_name)
                        transitions.append(transition)

        state.patterns.add(pattern_id)

        return pattern_id

    def match(self, expression):
        expression = freeze(expression)

        for state, subst in self._match(self.root, [expression], Substitution()):
            for pindex in state.patterns:
                renaming = self.pattern_vars[pindex]
                subst = subst.rename({renamed: original for original, renamed in renaming.items()})
                yield self.patterns[pindex], subst

    def _match(self, state, expressions, subst, associative=None):
        if not state.transitions:
            if state.patterns and not expressions:
                yield state, subst
            return

        if len(expressions) == 0 and OPERATION_END in state.transitions:
            yield state, subst

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
                                for matched_pattern, new_subst in matcher.match(expression.operands, subst):
                                    transition_set = transition.target.transitions[matched_pattern]
                                    for next_transition in transition_set:
                                        eventual_subst = self._check_constraint(next_transition, new_subst, matched_expr)
                                        if eventual_subst is not None:
                                            yield from self._match(next_transition.target, new_expressions, eventual_subst, associative)
                            else:
                                for new_state, new_subst in self._match(transition.target, expression.operands, subst, label if label.associative else None):
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
                                new_subst = self._check_constraint(transition, subst, matched_expr)
                                if new_subst is not None:
                                    yield from self._match(transition.target, new_expressions, new_subst, associative)
                            continue

                    if new_expressions is not None:
                        new_subst = self._check_constraint(transition, subst, matched_expr)
                        if new_subst is not None:
                            yield from self._match(transition.target, new_expressions, new_subst, associative)

    @staticmethod
    def _check_constraint(transition, subst, expr):
        if transition.variable_name is not None:
            try:
                subst = subst.union_with_variable(transition.variable_name, expr)
            except ValueError:
                return None
        if transition.constraint is None:
            return subst
        return subst if transition.constraint(subst) else None

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
                for pattern_id, subpatterns, variables in state.matcher.patterns.values():
                    var_formatted = ', '.join('{}[{}]x{}'.format(n, m, c) for n, c, m in variables)
                    submatch_label += '\n{}: {} {}'.format(pattern_id, subpatterns, var_formatted)
                dot.node(name+'-end', submatch_label, {'shape': 'box'})
                for f in subfinals:
                    dot.edge(f, name+'-end')
                dot.edge(name, 'n{}'.format(id(state.matcher.automaton.root)))
            elif not state.patterns:
                dot.node(name, '', {'shape': ('circle' if state.transitions else 'doublecircle' )})
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


class CommutativeMatcher(object):
    __slots__ = ('patterns', 'subjects', 'automaton', 'bipartite', 'associative')

    def __init__(self, associative):
        self.patterns = {}
        self.subjects = {}
        self.automaton = Automaton()
        self.bipartite = BipartiteGraph()
        self.associative = associative

    def add_pattern(self, pattern_id: int, operands: Iterable[FrozenExpression]) -> None:
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

        sorted_vars = tuple(sorted(pattern_vars.values()))
        sorted_subpatterns = tuple(sorted(pattern_set))
        pattern_key = sorted_subpatterns + sorted_vars
        if pattern_key not in self.patterns:
            inserted_id = len(self.patterns)
            self.patterns[pattern_key] = (inserted_id, pattern_set, sorted_vars)
        else:
            inserted_id = self.patterns[pattern_key][0]

        return inserted_id

    def _is_sequence_wildcard(self, wc):
        if isinstance(wc, Variable):
            wc = wc.expression
        if isinstance(wc, SymbolWildcard):
            return False
        if isinstance(wc, Wildcard):
            return not wc.fixed_size or self.associative
        return False

    def add_subject(self, subject: FrozenExpression) -> None:
        if subject not in self.subjects:
            subject_id = self.subjects[subject] = len(self.subjects)
            self.subjects[subject_id] = subject
            for state, parts in self.automaton._match(self.automaton.root, [subject], Substitution(), self.associative):
                for pattern_index in state.patterns:
                    variables = self.automaton.pattern_vars[pattern_index]
                    subst = Substitution((name, parts[index]) for name, index in variables.items())
                    self.bipartite[subject_id, pattern_index] = subst

    def match(self, subjects, substitution):
        subject_ids = Multiset(self.subjects[s] for s in subjects)
        for pattern_id, pattern_set, pattern_vars in self.patterns.values():
            if pattern_set:
                bipartite = self.build_bipartite(subject_ids, pattern_set)
                for matching in enum_maximum_matchings_iter(bipartite):
                    if len(matching) < len(pattern_set):
                        break
                    if not self._is_canonical_matching(matching, subject_ids, pattern_set):
                        continue
                    try:
                        subst = substitution.union(*(bipartite[edge] for edge in matching.items()))
                    except ValueError:
                        continue
                    if pattern_vars:
                        matched_expressions = Multiset(subexpression for subexpression, _ in matching)
                        remaining = Multiset(self.subjects[s] for s in (subject_ids - matched_expressions))
                        for sequence_subst in commutative_sequence_variable_partition_iter(remaining, pattern_vars):
                            try:
                                yield pattern_id, subst.union(sequence_subst)
                            except ValueError:
                                pass
                    elif len(subjects) == len(pattern_set):
                        yield pattern_id, subst
            elif pattern_vars:
                for subst in commutative_sequence_variable_partition_iter(Multiset(subjects), pattern_vars):
                    try:
                        yield pattern_id, substitution.union(subst)
                    except ValueError:
                        break
            elif len(subjects) == 0:
                yield pattern_id, substitution


    def build_bipartite(self, subjects, patterns):
        bipartite = BipartiteGraph()  # type: Subgraph
        for (expr, patt), m in self.bipartite.items():
            for i in range(subjects[expr]):
                for j in range(patterns[patt]):
                    bipartite[(expr, i), (patt, j)] = m
        return bipartite


    @staticmethod
    def _is_canonical_matching(matching, subject_ids, pattern_set):
        inverted_matching = {p: s for s, p in matching.items()}
        for (pattern_id, count) in sorted(pattern_set.items()):
            _, last = inverted_matching[pattern_id, 0]
            for i in range(1, count):
                _, j = inverted_matching[pattern_id, i]
                if j < last:
                    return False
                last = j
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
