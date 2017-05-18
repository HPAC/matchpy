# -*- coding: utf-8 -*-
from typing import Iterable, Iterator, List, Sequence, Tuple, cast, Set

from multiset import Multiset

from ..expressions.expressions import (
    Expression, Pattern, Operation, Symbol, SymbolWildcard, Wildcard, AssociativeOperation, CommutativeOperation
)
from ..expressions.constraints import Constraint
from ..expressions.substitution import Substitution
from ..expressions.functions import is_constant, preorder_iter_with_position, match_head, create_operation_expression
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
    if not is_constant(subject):
        raise ValueError("The subject for matching must be constant.")
    global_constraints = [c for c in pattern.constraints if not c.variables]
    local_constraints = set(c for c in pattern.constraints if c.variables)
    for subst in _match([subject], pattern.expression, Substitution(), local_constraints):
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
    if not is_constant(subject):
        raise ValueError("The subject for matching must be constant.")
    for child, pos in preorder_iter_with_position(subject):
        if match_head(child, pattern):
            for subst in match(child, pattern):
                yield subst, pos


def _match(subjects: List[Expression], pattern: Expression, subst: Substitution,
           constraints: Set[Constraint]) -> Iterator[Substitution]:
    match_iter = None
    expr = subjects[0] if subjects else None
    if isinstance(pattern, Wildcard):
        # All size checks are already handled elsewhere
        # When called directly from match, len(subjects) = 1
        # The operation matching also already only assigns valid number of subjects to a wildcard
        # So all we need to check here is the symbol type for SymbolWildcards
        if isinstance(pattern, SymbolWildcard) and not isinstance(subjects[0], pattern.symbol_type):
            return
        match_iter = iter([subst])
        if not pattern.fixed_size:
            expr = tuple(subjects)

    elif isinstance(pattern, Symbol):
        if len(subjects) == 1 and isinstance(subjects[0], type(pattern)) and subjects[0].name == pattern.name:
            match_iter = iter([subst])

    elif isinstance(pattern, Operation):
        if len(subjects) != 1 or not isinstance(subjects[0], pattern.__class__):
            return
        op_expr = cast(Operation, subjects[0])
        # if not op_expr.symbols >= pattern.symbols:
        #     return
        match_iter = _match_operation(op_expr, pattern, subst, _match, constraints)

    else:
        if len(subjects) == 1 and subjects[0] == pattern:
            match_iter = iter([subst])

    if match_iter is not None:
        if getattr(pattern, 'variable_name', False):
            for new_subst in match_iter:
                try:
                    new_subst = new_subst.union_with_variable(pattern.variable_name, expr)
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


def _match_factory(expressions, operand, constraints, matcher):
    def factory(subst):
        yield from matcher(expressions, operand, subst, constraints)

    return factory


def _count_seq_vars(expressions, operation):
    remaining = len(expressions)
    sequence_var_count = 0
    for operand in operation:
        if isinstance(operand, Wildcard):
            if not operand.fixed_size or isinstance(operation, AssociativeOperation):
                sequence_var_count += 1
            remaining -= operand.min_count
        else:
            remaining -= 1
        if remaining < 0:
            raise ValueError
    return remaining, sequence_var_count


def _build_full_partition(sequence_var_partition: Sequence[int], subjects: Sequence[Expression],
                          operation: Operation) -> List[Sequence[Expression]]:
    """Distribute subject operands among pattern operands.

    Given a partitoning for the variable part of the operands (i.e. a list of how many extra operands each sequence
    variable gets assigned).
    """
    i = 0
    var_index = 0
    result = []
    for operand in operation:
        wrap_associative = False
        if isinstance(operand, Wildcard):
            count = operand.min_count
            if not operand.fixed_size or isinstance(operation, AssociativeOperation):
                count += sequence_var_partition[var_index]
                var_index += 1
                wrap_associative = operand.fixed_size and operand.min_count
        else:
            count = 1

        operand_expressions = list(subjects)[i:i + count]
        i += count

        if wrap_associative and len(operand_expressions) > wrap_associative:
            fixed = wrap_associative - 1
            operand_expressions = tuple(operand_expressions[:fixed]) + (
                create_operation_expression(operation, operand_expressions[fixed:]),
            )

        result.append(operand_expressions)

    return result


def _non_commutative_match(subjects, operation, subst, constraints, matcher):
    try:
        remaining, sequence_var_count = _count_seq_vars(subjects, operation)
    except ValueError:
        return
    for part in weak_composition_iter(remaining, sequence_var_count):
        partition = _build_full_partition(part, subjects, operation)
        factories = [_match_factory(e, o, constraints, matcher) for e, o in zip(partition, operation)]

        for new_subst in generator_chain(subst, *factories):
            yield new_subst


def _match_operation(expressions, operation, subst, matcher, constraints):
    if len(operation) == 0:
        if len(expressions) == 0:
            yield subst
        return
    if not isinstance(operation, CommutativeOperation):
        yield from _non_commutative_match(expressions, operation, subst, constraints, matcher)
    else:
        parts = CommutativePatternsParts(type(operation), *operation)
        yield from _match_commutative_operation(expressions, parts, subst, constraints, matcher)


def _match_commutative_operation(
        subject_operands: Iterable[Expression],
        pattern: CommutativePatternsParts,
        substitution: Substitution,
        constraints,
        matcher
) -> Iterator[Substitution]:
    subjects = Multiset(subject_operands)  # type: Multiset
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
            if issubclass(pattern.operation, AssociativeOperation) and isinstance(replacement, pattern.operation):
                needed_count = Multiset(substitution[name])  # type: Multiset
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

    factories = [_fixed_expr_factory(e, constraints, matcher) for e in rest_expr]

    if not issubclass(pattern.operation, AssociativeOperation):
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
        if issubclass(pattern.operation, AssociativeOperation):
            sequence_vars += _variables_with_counts(fixed_vars, pattern.fixed_variable_infos)
            if pattern.wildcard_fixed is True:
                sequence_vars += (VariableWithCount(None, 1, pattern.wildcard_min_length), )
        if pattern.wildcard_fixed is False:
            sequence_vars += (VariableWithCount(None, 1, pattern.wildcard_min_length), )

        for sequence_subst in commutative_sequence_variable_partition_iter(Multiset(rem_expr), sequence_vars):
            if issubclass(pattern.operation, AssociativeOperation):
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


def _fixed_expr_factory(expression, constraints, matcher):
    def factory(data):
        expressions, substitution = data
        for expr in expressions.distinct_elements():
            if match_head(expr, expression):
                for subst in matcher([expr], expression, substitution, constraints):
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
