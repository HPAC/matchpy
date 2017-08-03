# -*- coding: utf-8 -*-
from typing import Iterable, Iterator, List, Sequence, Tuple, cast, Set

from multiset import Multiset

from ..expressions.expressions import (
    Expression, Pattern, Operation, Symbol, SymbolWildcard, Wildcard, AssociativeOperation, CommutativeOperation, OneIdentityOperation
)
from ..expressions.constraints import Constraint
from ..expressions.substitution import Substitution
from ..expressions.functions import (
    is_constant, preorder_iter_with_position, match_head, create_operation_expression, op_iter, op_len
)
from ..utils import (
    VariableWithCount, commutative_sequence_variable_partition_iter, fixed_integer_vector_iter, weak_composition_iter,
    generator_chain, optional_iter
)
from ._common import CommutativePatternsParts, check_one_identity

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
        if pattern.optional is not None and not subjects:
            expr = pattern.optional
        elif not pattern.fixed_size:
            expr = tuple(subjects)

    elif isinstance(pattern, Symbol):
        if len(subjects) == 1 and isinstance(subjects[0], type(pattern)) and subjects[0].name == pattern.name:
            match_iter = iter([subst])

    elif isinstance(pattern, Operation):
        if isinstance(pattern, OneIdentityOperation):
            yield from _match_one_identity(subjects, pattern, subst, constraints)
        if len(subjects) != 1 or not isinstance(subjects[0], pattern.__class__):
            return
        op_expr = cast(Operation, subjects[0])
        # if not op_expr.symbols >= pattern.symbols:
        #     return
        match_iter = _match_operation(op_expr, pattern, subst, constraints)

    else:
        if len(subjects) == 1 and subjects[0] == pattern:
            match_iter = iter([subst])

    if match_iter is not None:
        if getattr(pattern, 'variable_name', False):
            for new_subst in match_iter:
                try:
                    if expr is None and getattr(pattern, 'optional', None) is not None:
                        expr = pattern.optional
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


def _match_factory(subjects, operand, constraints):
    def factory(subst):
        yield from _match(subjects, operand, subst, constraints)

    return factory


def _count_seq_vars(subjects, operation):
    remaining = op_len(subjects)
    sequence_var_count = 0
    optional_count = 0
    for operand in op_iter(operation):
        if isinstance(operand, Wildcard):
            if not operand.fixed_size or isinstance(operation, AssociativeOperation):
                sequence_var_count += 1
                if operand.optional is None:
                    remaining -= operand.min_count
            elif operand.optional is not None:
                optional_count += 1
            else:
                remaining -= operand.min_count
        else:
            remaining -= 1
        if remaining < 0:
            raise ValueError
    return remaining, sequence_var_count, optional_count


def _build_full_partition(
        optional_parts, sequence_var_partition: Sequence[int], subjects: Sequence[Expression], operation: Operation
) -> List[Sequence[Expression]]:
    """Distribute subject operands among pattern operands.

    Given a partitoning for the variable part of the operands (i.e. a list of how many extra operands each sequence
    variable gets assigned).
    """
    i = 0
    var_index = 0
    opt_index = 0
    result = []
    for operand in op_iter(operation):
        wrap_associative = False
        if isinstance(operand, Wildcard):
            count = operand.min_count if operand.optional is None else 0
            if not operand.fixed_size or isinstance(operation, AssociativeOperation):
                count += sequence_var_partition[var_index]
                var_index += 1
                wrap_associative = operand.fixed_size and operand.min_count
            elif operand.optional is not None:
                count = optional_parts[opt_index]
                opt_index += 1
        else:
            count = 1

        operand_expressions = list(op_iter(subjects))[i:i + count]
        i += count

        if wrap_associative and len(operand_expressions) > wrap_associative:
            fixed = wrap_associative - 1
            operand_expressions = tuple(operand_expressions[:fixed]) + (
                create_operation_expression(operation, operand_expressions[fixed:]),
            )

        result.append(operand_expressions)

    return result


def _non_commutative_match(subjects, operation, subst, constraints):
    try:
        remaining, sequence_var_count, optional_count = _count_seq_vars(subjects, operation)
    except ValueError:
        return
    for new_remaining, optional in optional_iter(remaining, optional_count):
        if new_remaining < 0:
            continue
        for part in weak_composition_iter(new_remaining, sequence_var_count):
            partition = _build_full_partition(optional, part, subjects, operation)
            factories = [_match_factory(e, o, constraints) for e, o in zip(partition, op_iter(operation))]

            for new_subst in generator_chain(subst, *factories):
                yield new_subst


def _match_one_identity(subjects, operation, subst, constraints):
    non_optional, added_subst = check_one_identity(operation)
    if non_optional is not None:
        try:
            new_subst = subst.union(added_subst)
        except ValueError:
            return
        yield from _match(subjects, non_optional, new_subst, constraints)


def _match_operation(subjects, operation, subst, constraints):
    if op_len(operation) == 0:
        if op_len(subjects) == 0:
            yield subst
        return
    if not isinstance(operation, CommutativeOperation):
        yield from _non_commutative_match(subjects, operation, subst, constraints)
    else:
        parts = CommutativePatternsParts(type(operation), *op_iter(operation))
        yield from _match_commutative_operation(subjects, parts, subst, constraints)


def _match_commutative_operation(
        subject_operands: Iterable[Expression],
        pattern: CommutativePatternsParts,
        substitution: Substitution,
        constraints
) -> Iterator[Substitution]:
    subjects = Multiset(op_iter(subject_operands))  # type: Multiset
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
                needed_count = Multiset(op_iter(substitution[name]))  # type: Multiset
            else:
                if isinstance(replacement, (tuple, list, Multiset)):
                    return
                needed_count = Multiset({replacement: 1})
            if count > 1:
                needed_count *= count
            if not needed_count <= subjects:
                return
            subjects -= needed_count
            del fixed_vars[name]

    factories = [_fixed_expr_factory(e, constraints) for e in rest_expr]

    if not issubclass(pattern.operation, AssociativeOperation):
        for name, count in fixed_vars.items():
            min_count, symbol_type, default = pattern.fixed_variable_infos[name]
            factory = _fixed_var_iter_factory(name, count, min_count, symbol_type, constraints, default)
            factories.append(factory)

        if pattern.wildcard_fixed is True:
            factory = _fixed_var_iter_factory(None, 1, pattern.wildcard_min_length, None, constraints, None)
            factories.append(factory)
    else:
        for name, count in fixed_vars.items():
            min_count, symbol_type, default = pattern.fixed_variable_infos[name]
            if symbol_type is not None:
                factory = _fixed_var_iter_factory(name, count, min_count, symbol_type, constraints, default)
                factories.append(factory)

    for rem_expr, substitution in generator_chain((subjects, substitution), *factories):
        sequence_vars = _variables_with_counts(pattern.sequence_variables, pattern.sequence_variable_infos)
        if issubclass(pattern.operation, AssociativeOperation):
            sequence_vars += _variables_with_counts(fixed_vars, pattern.fixed_variable_infos)
            if pattern.wildcard_fixed is True:
                sequence_vars += (VariableWithCount(None, 1, pattern.wildcard_min_length, None), )
        if pattern.wildcard_fixed is False:
            sequence_vars += (VariableWithCount(None, 1, pattern.wildcard_min_length, None), )

        for sequence_subst in commutative_sequence_variable_partition_iter(Multiset(rem_expr), sequence_vars):
            if issubclass(pattern.operation, AssociativeOperation):
                for v in fixed_vars.distinct_elements():
                    if v not in sequence_subst:
                        continue
                    l = pattern.fixed_variable_infos[v].min_count
                    value = cast(Sequence, sequence_subst[v])
                    if isinstance(value, (list, tuple, Multiset)):
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
        VariableWithCount(name, count, infos[name].min_count, infos[name].default)
        for name, count in variables.items() if infos[name].type is None
    )


def _fixed_expr_factory(expression, constraints):
    def factory(data):
        subjects, substitution = data
        for expr in subjects.distinct_elements():
            if match_head(expr, expression):
                for subst in _match([expr], expression, substitution, constraints):
                    yield subjects - Multiset({expr: 1}), subst

    return factory


def _fixed_var_iter_factory(variable_name, count, length, symbol_type, constraints, optional):
    def factory(data):
        subjects, substitution = data
        if variable_name in substitution:
            value = ([substitution[variable_name]]
                     if not isinstance(substitution[variable_name], (tuple, list, Multiset)) else substitution[variable_name])
            if optional is not None and value == [optional]:
                yield subjects, substitution
            existing = Multiset(value) * count
            if not existing <= subjects:
                return
            yield subjects - existing, substitution
        else:
            if optional is not None:
                new_substitution = Substitution(substitution)
                new_substitution[variable_name] = optional
                yield subjects, new_substitution
            if length == 1:
                for expr, expr_count in subjects.items():
                    if expr_count >= count and (symbol_type is None or isinstance(expr, symbol_type)):
                        if variable_name is not None:
                            new_substitution = Substitution(substitution)
                            new_substitution[variable_name] = expr
                            for new_substitution in _check_constraints(new_substitution, constraints):
                                yield subjects - Multiset({expr: count}), new_substitution
                        else:
                            yield subjects - Multiset({expr: count}), substitution
            else:
                assert variable_name is None, "Fixed variables with length != 1 are not supported."
                exprs_with_counts = list(subjects.items())
                counts = tuple(c // count for _, c in exprs_with_counts)
                for subset in fixed_integer_vector_iter(counts, length):
                    sub_counter = Multiset(dict((exprs_with_counts[i][0], c * count) for i, c in enumerate(subset)))
                    yield subjects - sub_counter, substitution

    return factory
