# -*- coding: utf-8 -*-
import math
from typing import Callable, Iterator, List, Union, cast

from ..expressions import (Expression, Operation, Substitution, Symbol,
                           SymbolWildcard, Variable, Wildcard)
from ..utils import (commutative_partition_iter, integer_partition_vector_iter,
                     iterator_chain)


def match(expressions: List[Expression], pattern: Expression, subst: Substitution) -> Iterator[Substitution]:
    if isinstance(pattern, Variable):
        yield from _match_variable(expressions, pattern, subst, match)

    elif isinstance(pattern, Wildcard):
        yield from _match_wildcard(expressions, pattern, subst)

    elif isinstance(pattern, Symbol):
        if len(expressions) == 1 and expressions[0] == pattern:
            if pattern.constraint is None or pattern.constraint(subst):
                yield subst

    else:
        assert isinstance(pattern, Operation), "Unexpected expression of type {!r}".format(type(pattern))
        if len(expressions) != 1 or not isinstance(expressions[0], pattern.__class__):
            return
        op_expr = cast(Operation, expressions[0])
        for result in _match_operation(op_expr.operands, pattern, subst, match):
            if pattern.constraint is None or pattern.constraint(result):
                yield result

Matcher = Callable[[List[Expression], Expression, Substitution], Iterator[Substitution]]


def _match_variable(expressions: List[Expression], variable: Variable, subst: Substitution, matcher: Matcher) \
        -> Iterator[Substitution]:
    inner = variable.expression
    if len(expressions) == 1 and (not isinstance(inner, Wildcard) or inner.fixed_size):
        expr = expressions[0]  # type: Union[Expression, List[Expression]]
    else:
        expr = tuple(expressions)
    if variable.name in subst:
        if expr == subst[variable.name]:
            if variable.constraint is None or variable.constraint(subst):
                yield subst
        return
    for new_subst in matcher(expressions, variable.expression, subst):
        new_subst = new_subst.copy()
        new_subst[variable.name] = expr
        if variable.constraint is None or variable.constraint(new_subst):
            yield new_subst


def _match_wildcard(expressions: List[Expression], wildcard: Wildcard, subst: Substitution) -> Iterator[Substitution]:
    if wildcard.fixed_size:
        if len(expressions) == wildcard.min_count:
            if isinstance(wildcard, SymbolWildcard) and not isinstance(expressions[0], wildcard.symbol_type):
                return
            if wildcard.constraint is None or wildcard.constraint(subst):
                yield subst
    elif len(expressions) >= wildcard.min_count:
        if wildcard.constraint is None or wildcard.constraint(subst):
            yield subst


def _associative_operand_max(operand):
    while isinstance(operand, Variable):
        operand = operand.expression
    if isinstance(operand, Wildcard):
        return math.inf
    return 1


def _associative_fix_operand_max(parts, maxs, operation):
    new_parts = list(parts)
    for i, (part, max_count) in enumerate(zip(parts, maxs)):
        if len(part) > max_count:
            fixed = part[:max_count-1]
            variable = part[max_count-1:]
            new_parts[i] = tuple(fixed) + (operation.from_args(*variable), )
    return new_parts


def _size(expr):
    while isinstance(expr, Variable):
        expr = expr.expression
    if isinstance(expr, Wildcard):
        return (expr.min_count, (not expr.fixed_size) and math.inf or expr.min_count)
    return (1, 1)


def _partitions_iter(expressions, vars):
    integer_partition_vector_iter


def _match_factory(expressions, operand, matcher):
    def factory(subst):
        for subst in matcher(expressions, operand, subst):
            yield (subst, )

    return factory


def _count_seq_vars(expressions, operation):
    remaining = len(expressions)
    sequence_var_count = 0
    for operand in operation.operands:
        if isinstance(operand, Variable):
            operand = operand.expression
        if isinstance(operand, Wildcard):
            if not operand.fixed_size or operation.associative:
                sequence_var_count += 1
            remaining -= operand.min_count
        else:
            remaining -= 1
        if remaining < 0:
            raise ValueError
    return remaining, sequence_var_count


def _build_full_partition(sequence_var_partition, expressions, operation):
    i = 0
    var_index = 0
    result = []
    for operand in operation.operands:
        wrap_associative = False
        inner = operand.expression if isinstance(operand, Variable) else operand
        if isinstance(inner, Wildcard):
            count = inner.min_count
            if not inner.fixed_size or operation.associative:
                count += sequence_var_partition[var_index]
                var_index += 1
                wrap_associative = inner.fixed_size and inner.min_count
        else:
            count = 1

        operand_expressions = expressions[i:i+count]
        i += count

        if wrap_associative and len(operand_expressions) > wrap_associative:
            fixed = wrap_associative - 1
            op_factory = type(operation).from_args
            operand_expressions = tuple(operand_expressions[:fixed]) + (op_factory(*operand_expressions[fixed:]), )

        result.append(operand_expressions)

    return result


def _non_commutative_match(expressions, operation, subst, matcher):
    try:
        remaining, sequence_var_count = _count_seq_vars(expressions, operation)
    except ValueError:
        return

    for part in integer_partition_vector_iter(remaining, sequence_var_count):
        partition = _build_full_partition(part, expressions, operation)
        factories = [_match_factory(e, o, matcher) for e, o in zip(partition, operation.operands)]

        for (new_subst, ) in iterator_chain((subst, ), *factories):
            yield new_subst


def _match_operation(expressions, operation, subst, matcher):
    if len(operation.operands) == 0:
        if len(expressions) == 0:
            yield subst
        return

    if not operation.commutative:
        yield from _non_commutative_match(expressions, operation, subst, matcher)
        return

    # TODO
    mins, maxs = map(list, zip(*map(_size, operation.operands)))
    if operation.associative:
        fake_maxs = list(_associative_operand_max(o) for o in operation.operands)
    else:
        fake_maxs = maxs
    if len(expressions) < sum(mins) or len(expressions) > sum(fake_maxs):
        return
    parts = commutative_partition_iter(expressions, mins, fake_maxs)

    for part in parts:
        if operation.associative:
            part = _associative_fix_operand_max(part, maxs, type(operation))
        o_count = len(operation.operands)
        iterators = [None] * o_count
        next_subst = subst
        i = 0
        while True:
            try:
                while i < o_count:
                    if iterators[i] is None:
                        iterators[i] = matcher(part[i], operation.operands[i], next_subst)
                    next_subst = iterators[i].__next__()
                    i += 1
                yield next_subst
                i -= 1
            except StopIteration:
                iterators[i] = None
                i -= 1
                if i < 0:
                    break
