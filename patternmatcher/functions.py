# -*- coding: utf-8 -*-
import itertools
import math
from typing import (Callable, Iterator, List, NamedTuple, Sequence, Tuple,
                    Union, cast)

from patternmatcher.expressions import (Arity, Expression, Operation,
                                        Substitution, Symbol, SymbolWildcard,
                                        Variable, Wildcard)
from patternmatcher.utils import (commutative_partition_iter,
                                  integer_partition_vector_iter,
                                  iterator_chain)


def match(expression: Expression, pattern: Expression) -> Iterator[Substitution]:
    r"""Tries to match the given `pattern` to the given `expression`.

    Yields each match in form of a substitution that when applied to `pattern` results in the original
    `expression`.

    Parameters:
        expression:
            An expression to match.
        pattern:
            The pattern to match.

    Yields:
        All possible substitutions as dictionaries where each key is the name of the variable
        and the corresponding value is the variables substitution. Applying the substitution to the pattern
        results in the original expression (except for :class:`Wildcard`\s)
    """
    return _match([expression], pattern, {})


def match_anywhere(expression: Expression, pattern: Expression) -> Iterator[Tuple[Substitution, Tuple[int, ...]]]:
    """Tries to match the given `pattern` to the any subexpression of the given `expression`.

    Yields each match in form of a substitution and a position tuple.
    The substitution is a dictionary, where the key is the name of a variable and the value either an expression
    or a list of expressins (iff the variable is a sequence variable).
    When applied to `pattern`, the substitution results in the original matched subexpression.
    The position is a tuple of indices, e.g. the empty tuple refers to the `expression` itself,
    `(0, )` refers to the first child (operand) of the expression, `(0, 0)` to the first child of
    the first child etc.

    Parameters:
        expression:
            An expression to match.
        pattern:
            The pattern to match.

    Yields:
        All possible substitution and position pairs.
    """
    predicate = None
    if pattern.head is not None:
        predicate = lambda x: x.head == pattern.head
    for child, pos in expression.preorder_iter(predicate):
        for subst in _match([child], pattern, {}):
            yield subst, pos


def _match(expressions: List[Expression], pattern: Expression, subst: Substitution) -> Iterator[Substitution]:
    if isinstance(pattern, Variable):
        yield from _match_variable(expressions, pattern, subst, _match)

    elif isinstance(pattern, Wildcard):
        yield from _match_wildcard(expressions, pattern, subst)

    elif isinstance(pattern, Symbol):
        if len(expressions) == 1 and expressions[0] == pattern:
            if pattern.constraint is None or pattern.constraint(subst):
                yield subst

    else:
        assert isinstance(pattern, Operation), 'Unexpected expression of type %r' % type(pattern)
        if len(expressions) != 1 or not isinstance(expressions[0], pattern.__class__):
            return
        op_expr = cast(Operation, expressions[0])
        for result in _match_operation(op_expr.operands, pattern, subst, _match):
            if pattern.constraint is None or pattern.constraint(result):
                yield result

Matcher = Callable[[List[Expression], Expression, Substitution], Iterator[Substitution]]


def _match_variable(expressions: List[Expression], variable: Variable, subst: Substitution, matcher: Matcher) -> Iterator[Substitution]:
    inner = variable.expression
    if len(expressions) == 1 and (not isinstance(inner, Wildcard) or inner.fixed_size):
        expr = expressions[0]  # type: Union[Expression,List[Expression]]
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
    limits = list(zip(mins, fake_maxs))
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


def substitute(expression: Expression, substitution: Substitution) -> Tuple[Union[Expression, List[Expression]], bool]:
    """Replaces variables in the given `expression` by the given `substitution`.

    In addition to the resulting expression(s), a bool is returned indicating whether anything was substituted.
    If nothing was substituted, the original expression is returned.
    Not that this function returns a list of expressions iff the expression is a variable and its substitution
    is a list of expressions. In other cases were a substitution is a list of expressions, the expressions will
    be integrated as operands in the surrounding operation:

    >>> substitute(f(x_, c), {'x': [a, b]})
    (f(Symbol('a'), Symbol('b'), Symbol('c')), True)

    Parameters:
        expression:
            An expression in which variables are substituted.
        substitution:
            A substitution dictionary. The key is the name of the variable,
            the value either an expression or a list of expression to use as a replacement for
            the variable.

    Returns:
        The expression resulting from applying the substitution.
    """
    if isinstance(expression, Variable):
        if expression.name in substitution:
            return substitution[expression.name], True
    elif isinstance(expression, Operation):
        any_replaced = False
        new_operands = []
        for operand in expression.operands:
            result, replaced = substitute(operand, substitution)
            if replaced:
                any_replaced = True
            if isinstance(result, list):
                new_operands.extend(result)
            else:
                new_operands.append(result)
        if any_replaced:
            return type(expression).from_args(*new_operands), True

    return expression, False


def replace(expression: Expression, position: Sequence[int], replacement: Union[Expression, List[Expression]]) -> Union[Expression, List[Expression]]:
    r"""Replaces the subexpression of `expression` at the given `position` with the given `replacement`.

    The original `expression` itself is not modified, but a modified copy is returned. If the replacement
    is a list of expressions, it will be expanded into the list of operands of the respective operation:

    >>> replace(f(a), (0, ), [b, c])
    f(Symbol('b'), Symbol('c'))

    Parameters:
        expression:
            An :class:`Expression` where a (sub)expression is to be replaced.
        position:
            A tuple of indices, e.g. the empty tuple refers to the `expression` itself,
            `(0, )` refers to the first child (operand) of the `expression`, `(0, 0)` to the first
            child of the first child etc.
        replacement:
            Either an :class:`Expression` or a list of :class:`Expression`\s to be
            inserted into the `expression` instead of the original expression at that `position`.

    Returns:
        The resulting expression from the replacement.
    """
    if position == ():
        return replacement
    if not isinstance(expression, Operation):
        raise IndexError('Invalid position %r for expression %s' % (position, expression))
    if position[0] >= len(expression.operands):
        raise IndexError('Position %r out of range for expression %s' % (position, expression))
    op_class = type(expression)
    pos = position[0]
    subexpr = replace(expression.operands[pos], position[1:], replacement)
    if isinstance(subexpr, list):
        return op_class.from_args(*(expression.operands[:pos] + subexpr + expression.operands[pos+1:]))
    operands = expression.operands.copy()
    operands[pos] = subexpr
    return op_class.from_args(*operands)

ReplacementRule = NamedTuple('ReplacementRule', [('pattern', Expression), ('replacement', Callable[..., Expression])])


def replace_all(expression: Expression, rules: Sequence[ReplacementRule]) -> Union[Expression, List[Expression]]:
    grouped = itertools.groupby(rules, lambda r: r.pattern.head)
    heads, tmp_groups = map(list, zip(*[(h, list(g)) for h, g in grouped]))
    groups = [list(g) for g in tmp_groups]
    replaced = True
    while replaced:
        replaced = False
        for head, group in zip(heads, groups):
            predicate = None
            if head is not None:
                predicate = lambda e: e.head == head
            for subexpr, pos in expression.preorder_iter(predicate):
                for pattern, replacement in group:
                    try:
                        subst = next(match(subexpr, pattern))
                        result = replacement(**subst)
                        expression = replace(expression, pos, result)
                        replaced = True
                        break
                    except StopIteration:
                        pass
                if replaced:
                    break
            if replaced:
                break

    return expression

if __name__ == '__main__':
    import doctest

    f = Operation.new('f', Arity.variadic)
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    x_ = Variable.dot('x')

    doctest.testmod(exclude_empty=True)
