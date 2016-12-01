# -*- coding: utf-8 -*-
from collections import deque

from graphviz import Digraph

from ..expressions import Operation, Variable, Wildcard, Symbol, SymbolWildcard, freeze, Substitution
from .syntactic import OPERATION_END, is_operation, is_symbol_wildcard, FlatTerm
from ..constraints import MultiConstraint, EqualVariablesConstraint

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

class State:
    def __init__(self):
        self.transitions = {}
        self.patterns = set()

    def __repr__(self):
        return 'State({!r}, {!r})'.format(self.transitions, self.patterns)

class Transition:
    def __init__(self, label, target, constraint):
        self.label = label
        self.target = target
        self.constraint = constraint
        self.same_var_indices = set()

class Automaton:
    def __init__(self):
        self.patterns = []
        self.states = []
        self.root = self._create_state()
        self.pattern_vars = []

    def _create_state(self):
        state = State()
        self.states.append(state)
        return state

    def add(self, pattern):
        pattern_id = len(self.patterns)
        pattern_vars = {}
        index = 0
        self.patterns.append(pattern)
        self.pattern_vars.append(pattern_vars)
        pattern = freeze(pattern)
        state = self.root
        expressions_stack = [deque([pattern])]
        context_stack = []
        transition = None

        while expressions_stack:
            if expressions_stack[-1]:
                expr = expressions_stack[-1].popleft()
                label = None
                same_var_index = None
                constraint = None
                if isinstance(expr, Variable):
                    constraint = expr.constraint
                    if expr.name in pattern_vars:
                        same_var_index = pattern_vars[expr.name]
                    else:
                        pattern_vars[expr.name] = index
                    expr = expr.expression
                if isinstance(expr, Operation):
                    label = type(expr)
                    expressions_stack.append(deque(expr.operands))
                    context_stack.append((MultiConstraint.create(constraint, expr.constraint), same_var_index))
                else:
                    label = expr.without_constraints
                renaming = dict((n, 'i{}'.format(i)) for n, i in pattern_vars.items())
                constraint = MultiConstraint.create(constraint, expr.constraint)
                if constraint is not None:
                    constraint = constraint.with_renamed_vars(renaming)
                if label is not None:
                    if (label, constraint) in state.transitions:
                        transition = state.transitions[(label, constraint)]
                        state = transition.target
                    else:
                        new_state = self._create_state()
                        transition = Transition(label, new_state, constraint)
                        state.transitions[(label, constraint)] = transition
                        state = new_state
                    if same_var_index is not None:
                        transition.same_var_indices.add(same_var_index)
                index += 1
            else:
                expressions_stack.pop()
                if context_stack:
                    constraint, same_var_index = context_stack.pop()
                    if (OPERATION_END, constraint) in state.transitions:
                        transition = state.transitions[(OPERATION_END, constraint)]
                        state = transition.target
                    else:
                        new_state = self._create_state()
                        transition = Transition(OPERATION_END, new_state, constraint)
                        state.transitions[(OPERATION_END, constraint)] = transition
                        state = new_state
                    if same_var_index is not None:
                        transition.same_var_indices.add(same_var_index)

        state.patterns.add(pattern_id)

    def match(self, expression):
        expression = freeze(expression)

        for state, parts in self._match(self.root, [expression], []):
            for pindex in state.patterns:
                subst = Substitution((name, parts[index]) for name, index in self.pattern_vars[pindex].items())
                yield self.patterns[pindex], subst

    def _match(self, state, expressions, parts, associative=None):
        if not state.transitions:
            if state.patterns and not expressions:
                yield state, parts
            return

        if len(expressions) == 0 and any(l == OPERATION_END for (l, _) in state.transitions):
            yield state, parts

        expression = expressions[0] if expressions else None

        for (label, constraint), transition in state.transitions.items():
            new_parts = parts[:]
            new_expressions = None
            if is_operation(label):
                if isinstance(expression, label):
                    new_parts = parts + [expression]
                    new_expressions = expressions[1:]
                    for new_state, new_parts2 in self._match(transition.target, expression.operands, new_parts, label if label.associative else None):
                        for (label2, constraint2), transition in new_state.transitions.items():
                            if label2 == OPERATION_END and self._check_constraint(constraint2, new_parts2, transition.same_var_indices):
                                yield from self._match(transition.target, new_expressions, new_parts2, associative)
                    continue
            elif isinstance(label, Symbol):
                if label == expression:
                    new_parts = parts + [expression]
                    new_expressions = expressions[1:]
            elif isinstance(label, SymbolWildcard):
                if isinstance(expression, label.symbol_type):
                    new_parts = parts + [expression]
                    new_expressions = expressions[1:]
            elif isinstance(label, Wildcard):
                min_count = label.min_count
                if label.fixed_size and not associative:
                    if min_count == 1:
                        if expression is not None:
                            new_parts = parts + [expression]
                            new_expressions = expressions[1:]
                    elif len(expressions) >= min_count:
                        new_parts = parts + [tuple(expressions[:min_count])]
                        new_expressions = expressions[min_count:]
                else:
                    for i in range(min_count, len(expressions) + 1):
                        new_part = tuple(expressions[:i])
                        if associative and label.fixed_size:
                            if i > min_count:
                                wrapped = associative.from_args(*new_part[min_count-1:])
                                if min_count == 1:
                                    new_part = wrapped
                                else:
                                    new_part = new_part[:min_count-1] + (wrapped, )
                            elif min_count == 1:
                                new_part = new_part[0]
                        new_parts = parts + [new_part]
                        new_expressions = expressions[i:]
                        if self._check_constraint(constraint, new_parts, transition.same_var_indices):
                            yield from self._match(transition.target, new_expressions, new_parts, associative)
                    continue

            if new_expressions is not None:

                if self._check_constraint(constraint, new_parts, transition.same_var_indices):
                    yield from self._match(transition.target, new_expressions, new_parts, associative)

    @staticmethod
    def _check_constraint(constraint, parts, same_var_indices=[]):
        for index in same_var_indices:
            if parts[-1] != parts[index]:
                return False
        if constraint is None:
            return True
        subst = Substitution(('i{}'.format(i), p) for i, p in enumerate(parts))
        result = constraint(subst)
        return result

    def as_graph(self) -> Digraph:  # pragma: no cover
        dot = Digraph()

        for state in self.states:
            if not state.patterns:
                dot.node('n{!s}'.format(id(state)), '', {'shape': ('circle' if state.transitions else 'doublecircle' )})
            else:
                vars = ['{}: {}'.format(p, repr(self.pattern_vars[p])) for p in state.patterns]
                label = '\n'.join(vars)
                dot.node('n{!s}'.format(id(state)), label, {'shape': 'box'})

        for state in self.states:
            for ((label, constraint), transition) in state.transitions.items():
                str_label = str(label)
                if constraint is not None:
                    str_label += ' ;/ ' + str(constraint)
                if transition.same_var_indices:
                    str_label += '\n \\== {}'.format(' \\== '.join(map(str, transition.same_var_indices)))

                dot.edge('n{!s}'.format(id(state)), 'n{!s}'.format(id(transition.target)), str_label)

        return dot
