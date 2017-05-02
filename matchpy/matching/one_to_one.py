# -*- coding: utf-8 -*-
import math
from typing import Iterable, Iterator, List, Sequence, Tuple, cast, Set, Optional, Type
from itertools import chain

from multiset import Multiset

from ..expressions import Expression, Pattern, Operation, Symbol, SymbolWildcard, Wildcard, Alternatives, ExpressionSequence, Repeated
from ..expressions.constraints import Constraint
from ..expressions.substitution import Substitution
from ..utils import (
    VariableWithCount, commutative_sequence_variable_partition_iter, fixed_integer_vector_iter, weak_composition_iter,
    generator_chain
)
from ._common import CommutativePatternsParts

__all__ = ['match', 'match_anywhere']


def match(subject: Expression, pattern: Pattern) -> Iterator[Substitution]:
    r"""Tries to match the given *pattern* to the given *subject*.

    Yields each match in form of a substitution.

    Parameters:
        subject:
            An subject to match.
        pattern:
            The pattern to match.

    Yields:
        All possible match substitutions.

    Raises:
        ValueError:
            If the subject is not constant.
    """
    if not subject.is_constant:
        raise ValueError("The subject for matching must be constant.")
    global_constraints = [c for c in pattern.constraints if not c.variables]
    local_constraints = set(c for c in pattern.constraints if c.variables)
    for subst in _match([subject], pattern.expression, Substitution(), local_constraints, None):
        for constraint in global_constraints:
            if not constraint(subst):
                break
        else:
            yield subst


def match_anywhere(subject: Expression, pattern: Pattern) -> Iterator[Tuple[Substitution, Tuple[int, ...]]]:
    """Tries to match the given *pattern* to the any subexpression of the given *subject*.

    Yields each match in form of a substitution and a position tuple.
    The position is a tuple of indices, e.g. the empty tuple refers to the *subject* itself,
    :code:`(0, )` refers to the first child (operand) of the subject, :code:`(0, 0)` to the first child of
    the first child etc.

    Parameters:
        subject:
            An subject to match.
        pattern:
            The pattern to match.

    Yields:
        All possible substitution and position pairs.

    Raises:
        ValueError:
            If the subject is not constant.
    """
    if not subject.is_constant:
        raise ValueError("The subject for matching must be constant.")
    head = pattern.expression.head
    if head is None:
        child_iterator = subject.preorder_iter()
    else:
        child_iterator = subject.preorder_iter(lambda child: child.head == head)
    for child, pos in child_iterator:
        for subst in match(child, pattern):
            yield subst, pos


def _match(
        subjects: List[Expression],
        pattern: Expression,
        subst: Substitution,
        constraints: Set[Constraint],
        associative: Optional[Type[Operation]]
) -> Iterator[Substitution]:
    match_iter = None
    subject = subjects[0] if len(subjects) == 1 else None
    if isinstance(pattern, Wildcard):
        # All size checks are already handled elsewhere
        # When called directly from match, len(subjects) = 1
        # The operation matching also already only assigns valid number of subjects to a wildcard
        # So all we need to check here is the symbol type for SymbolWildcards
        if isinstance(pattern, SymbolWildcard) and not isinstance(subjects[0], pattern.symbol_type):
            return
        match_iter = iter([subst])
        if not pattern.fixed_size:
            subject = tuple(subjects)
        elif associative and len(subjects) > 1:
            subject = associative(*subjects)

    elif isinstance(pattern, Symbol):
        if len(subjects) == 1 and isinstance(subjects[0], type(pattern)) and subjects[0].name == pattern.name:
            match_iter = iter([subst])

    elif isinstance(pattern, Operation):
        if len(subjects) != 1 or not isinstance(subjects[0], pattern.__class__):
            return
        op_expr = cast(Operation, subjects[0])
        if not op_expr.symbols >= pattern.symbols:
            return
        subjects = op_expr.operands
        if len(pattern.operands) == 0:
            if len(subjects) == 0:
                yield subst
            return
        if not op_expr.commutative:
            associative = op_expr.associative and type(op_expr)
            match_iter = _match_sequence(subjects, pattern.operands, subst, constraints, associative)
        else:
            parts = CommutativePatternsParts(type(op_expr), *pattern.operands)
            match_iter = _match_commutative_sequence(subjects, parts, subst, constraints)

    elif isinstance(pattern, Alternatives):
        match_iter = chain.from_iterable(
            _match(subjects, option, subst, constraints, associative) for option in pattern.children
        )

    elif isinstance(pattern, ExpressionSequence):
        match_iter = _match_sequence(subjects, pattern.children, subst, constraints, associative)

    elif isinstance(pattern, Repeated):
        min_repeats = max(int(len(subjects) / pattern.expression.max_length), pattern.min_count)
        max_repeats = min(int(len(subjects) / pattern.expression.min_length), pattern.max_count)

        match_iter = chain.from_iterable(
            _match_sequence(subjects, [pattern.expression] * i, subst, constraints, associative)
            for i in range(min_repeats, max_repeats + 1)
        )

    else:
        assert False, "Unexpected pattern of type {!r}".format(type(pattern))

    if match_iter is not None:
        if pattern.variable_name:
            for new_subst in match_iter:
                try:
                    new_subst = new_subst.union_with_variable(pattern.variable_name, subject)
                except ValueError:
                    pass
                else:
                    yield from _check_constraints(new_subst, constraints)
        else:
            yield from match_iter


def _check_constraints(substitution, constraints):
    restore_constraints = set()
    try:
        for constraint in list(constraints):
            for var in constraint.variables:
                if var not in substitution:
                    break
            else:
                if not constraint(substitution):
                    break
                restore_constraints.add(constraint)
                constraints.remove(constraint)
        else:
            yield substitution
    finally:
        for constraint in restore_constraints:
            constraints.add(constraint)


def _match_factory(expressions, operand, constraints, associative):
    def factory(subst):
        yield from _match(expressions, operand, subst, constraints, associative)

    return factory


def _count_seq_vars(subject_count, patterns, associative):
    remaining = subject_count
    sequence_var_count = 0
    # TODO: regular vars in associative operation need to be handled correctly
    for operand in patterns:
        if isinstance(operand, Wildcard):
            if not operand.fixed_size or associative:
                sequence_var_count += 1
            remaining -= operand.min_count
        elif isinstance(operand, ExpressionSequence):
            sequence_var_count += 1
            remaining -= operand.min_length
        elif isinstance(operand, Alternatives):
            if operand.min_length != 1 or operand.max_length != 1:
                sequence_var_count += 1
            remaining -= operand.min_length
        elif isinstance(operand, Repeated):
            sequence_var_count += 1
            remaining -= operand.min_length
        else:
            remaining -= 1
        if remaining < 0:
            raise ValueError
    return remaining, sequence_var_count


def _is_wrapped_sequence_var(expression, associative):
    return associative and isinstance(expression, Wildcard) and expression.min_count == 1 and expression.fixed_size


def _build_full_partition(
        sequence_var_partition: Sequence[int],
        subjects: Sequence[Expression],
        patterns: Sequence[Expression],
        associative: Optional[Type[Operation]]
) -> List[Sequence[Expression]]:
    """Distribute subject operands among pattern operands.

    Given a partitoning for the variable part of the operands (i.e. a list of how many extra operands each sequence
    variable gets assigned).
    """
    i = 0
    var_index = 0
    result = []
    for operand in patterns:
        wrap_associative = False
        count = operand.min_length
        if count != 1 or operand.max_length != operand.min_length or _is_wrapped_sequence_var(operand, associative):
            count += sequence_var_partition[var_index]
            var_index += 1
            if isinstance(operand, Wildcard) and (not operand.fixed_size or associative):
                wrap_associative = operand.fixed_size and operand.min_count

        operand_expressions = subjects[i:i + count]
        i += count

        if wrap_associative and len(operand_expressions) > wrap_associative:
            fixed = wrap_associative - 1
            operand_expressions = tuple(operand_expressions[:fixed]) + (associative(*operand_expressions[fixed:]), )

        result.append(operand_expressions)

    return result


def _match_sequence(subjects, patterns, subst, constraints, associative):
    try:
        remaining, sequence_var_count = _count_seq_vars(len(subjects), patterns, associative)
    except ValueError:
        return
    for part in weak_composition_iter(remaining, sequence_var_count):
        partition = _build_full_partition(part, subjects, patterns, associative)
        factories = [_match_factory(e, o, constraints, associative) for e, o in zip(partition, patterns)]

        for new_subst in generator_chain(subst, *factories):
            yield new_subst


def _match_commutative_sequence(
        subjects: Iterable[Expression], pattern: CommutativePatternsParts, substitution: Substitution, constraints
) -> Iterator[Substitution]:
    subjects = Multiset(subjects)  # type: Multiset
    if not pattern.constant <= subjects:
        return
    subjects -= pattern.constant
    rest_expr = pattern.rest + pattern.syntactic
    needed_length = (
        pattern.sequence_variable_min_length + pattern.fixed_variable_length + len(rest_expr) +
        pattern.wildcard_min_length
    )

    if len(subjects) < needed_length:
        return

    fixed_vars = Multiset(pattern.fixed_variables)  # type: Multiset[str]
    for name, count in pattern.fixed_variables.items():
        if name in substitution:
            replacement = substitution[name]
            if pattern.operation.associative and isinstance(replacement, pattern.operation):
                needed_count = Multiset(cast(Operation, substitution[name]).operands)  # type: Multiset
            else:
                if not isinstance(replacement, Expression):
                    return
                needed_count = Multiset({replacement: 1})
            if count > 1:
                needed_count *= count
            if not needed_count <= subjects:
                return
            subjects -= needed_count
            del fixed_vars[name]

    associative = pattern.operation if pattern.operation.associative else None
    factories = [_fixed_expr_factory(e, constraints, associative) for e in rest_expr]

    if not pattern.operation.associative:
        for name, count in fixed_vars.items():
            min_count, symbol_type = pattern.fixed_variable_infos[name]
            factory = _fixed_var_iter_factory(name, count, min_count, symbol_type, constraints)
            factories.append(factory)

        if pattern.wildcard_fixed is True:
            factory = _fixed_var_iter_factory(None, 1, pattern.wildcard_min_length, None, constraints)
            factories.append(factory)
    else:
        for name, count in fixed_vars.items():
            min_count, symbol_type = pattern.fixed_variable_infos[name]
            if symbol_type is not None:
                factory = _fixed_var_iter_factory(name, count, min_count, symbol_type, constraints)
                factories.append(factory)

    expr_counter = Multiset(subjects)  # type: Multiset

    for rem_expr, substitution in generator_chain((expr_counter, substitution), *factories):
        sequence_vars = _variables_with_counts(pattern.sequence_variables, pattern.sequence_variable_infos)
        if pattern.operation.associative:
            sequence_vars += _variables_with_counts(fixed_vars, pattern.fixed_variable_infos)
            if pattern.wildcard_fixed is True:
                sequence_vars += (VariableWithCount(None, 1, pattern.wildcard_min_length), )
        if pattern.wildcard_fixed is False:
            sequence_vars += (VariableWithCount(None, 1, pattern.wildcard_min_length), )

        for sequence_subst in commutative_sequence_variable_partition_iter(Multiset(rem_expr), sequence_vars):
            if pattern.operation.associative:
                for v in fixed_vars.distinct_elements():
                    if v not in sequence_subst:
                        continue
                    l = pattern.fixed_variable_infos[v].min_count
                    value = cast(Multiset, sequence_subst[v])
                    if len(value) > l:
                        normal = Multiset(list(value)[:l - 1])
                        wrapped = pattern.operation(*(value - normal))
                        normal.add(wrapped)
                        sequence_subst[v] = normal if l > 1 else next(iter(normal))
                    else:
                        assert len(value) == 1 and l == 1, "Fixed variables with length != 1 are not supported."
                        sequence_subst[v] = next(iter(value))
            try:
                result = substitution.union(sequence_subst)
            except ValueError:
                pass
            else:
                yield from _check_constraints(result, constraints)


def _variables_with_counts(variables, infos):
    return tuple(
        VariableWithCount(name, count, infos[name].min_count)
        for name, count in variables.items() if infos[name].type is None
    )


def _fixed_expr_factory(expression, constraints, associative):
    def factory(data):
        expressions, substitution = data
        for expr in expressions.distinct_elements():
            if expr.head == expression.head:
                for subst in _match([expr], expression, substitution, constraints, associative):
                    yield expressions - Multiset({expr: 1}), subst

    return factory


def _fixed_var_iter_factory(variable_name, count, length, symbol_type, constraints):
    def factory(data):
        expressions, substitution = data
        if variable_name in substitution:
            value = ([substitution[variable_name]]
                     if isinstance(substitution[variable_name], Expression) else substitution[variable_name])
            existing = Multiset(value) * count
            if not existing <= expressions:
                return
            yield expressions - existing, substitution
        else:
            if length == 1:
                for expr, expr_count in expressions.items():
                    if expr_count >= count and (symbol_type is None or isinstance(expr, symbol_type)):
                        if variable_name is not None:
                            new_substitution = Substitution(substitution)
                            new_substitution[variable_name] = expr
                            for new_substitution in _check_constraints(new_substitution, constraints):
                                yield expressions - Multiset({expr: count}), new_substitution
                        else:
                            yield expressions - Multiset({expr: count}), substitution
            else:
                assert variable_name is None, "Fixed variables with length != 1 are not supported."
                exprs_with_counts = list(expressions.items())
                counts = tuple(c // count for _, c in exprs_with_counts)
                for subset in fixed_integer_vector_iter(counts, length):
                    sub_counter = Multiset(dict((exprs_with_counts[i][0], c * count) for i, c in enumerate(subset)))
                    yield expressions - sub_counter, substitution

    return factory
