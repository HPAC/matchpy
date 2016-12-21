# -*- coding: utf-8 -*-
"""This module contains various functions for working with expressions.

- With `substitute()` you can replace occurrences of variables with an expression or sequence of expressions.
- With `replace()` you can replace a subexpression at a specific position with a differen expression or
  sequence of expressions.
- With `replace_all()` you can apply a set of replacement rules repeatedly to an expression.
- With `is_match()` you can check whether a pattern matches a subject expression.
"""
import itertools
import math
from typing import Callable, List, NamedTuple, Sequence, Tuple, Union, Iterable

from multiset import Multiset

from .expressions import Expression, Operation, Substitution, Variable, freeze
from .matching.one_to_one import match

__all__ = ['substitute', 'replace', 'replace_all', 'is_match']


def substitute(expression: Expression, substitution: Substitution) -> Tuple[Union[Expression, List[Expression]], bool]:
    """Replaces variables in the given `expression` by the given `substitution`.

    >>> print(substitute(f(x_), {'x': a})[0])
    f(a)

    In addition to the resulting expression(s), a bool is returned indicating whether anything was substituted.
    If nothing was substituted, the original expression is returned:

    >>> result, was_substituted = substitute(f(x_), {'y': a})
    >>> print(result, was_substituted)
    f(x_) False

    Note that this function returns a list of expressions iff the expression is a variable and its substitution
    is a list of expressions. In other cases were a substitution is a list of expressions, the expressions will
    be integrated as operands in the surrounding operation:

    >>> print(substitute(f(x_, c), {'x': [a, b]})[0])
    f(a, b, c)

    If you substitute with a `Multiset` of values, they will be sorted:

    >>> replacement = Multiset(map(freeze, [b, a, b]))
    >>> print(substitute(f(x_, c), {'x': replacement})[0])
    f(a, b, b, c)

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
            if isinstance(result, Expression):
                new_operands.append(result)
            elif isinstance(result, Multiset):
                new_operands.extend(sorted(result))
            else:
                new_operands.extend(result)
        if any_replaced:
            return type(expression).from_args(*new_operands), True

    return expression, False


def replace(expression: Expression, position: Sequence[int], replacement: Union[Expression, Sequence[Expression]]) \
        -> Union[Expression, Sequence[Expression]]:
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

    Raises:
        IndexError: If the position is invalid or out of range.
    """
    if position == ():
        return replacement
    if not isinstance(expression, Operation):
        raise IndexError("Invalid position {!r} for expression {!s}".format(position, expression))
    if position[0] >= len(expression.operands):
        raise IndexError("Position {!r} out of range for expression {!s}".format(position, expression))
    op_class = type(expression)
    pos = position[0]
    subexpr = replace(expression.operands[pos], position[1:], replacement)
    if isinstance(subexpr, Sequence):
        new_operands = tuple(expression.operands[:pos]) + tuple(subexpr) + tuple(expression.operands[pos + 1:])
        return op_class.from_args(*new_operands)
    operands = list(expression.operands)
    operands[pos] = subexpr
    return op_class.from_args(*operands)


ReplacementRule = NamedTuple('ReplacementRule', [('pattern', Expression), ('replacement', Callable[..., Expression])])


def replace_all(expression: Expression, rules: Iterable[ReplacementRule], max_count: int=math.inf) \
        -> Union[Expression, Sequence[Expression]]:
    """Replace all occurrences of the patterns according to the replacement rules.

    A replacement rule consists of a *pattern*, that is matched against any subexpression
    of the expression. If a match is found, the *replacement* callback of the rule is called with
    the variables from the match substitution. Whatever the callback returns is used as a replacement for the
    matched subexpression. This can either be a single expression or a sequence of expressions, which is then
    integrated into the surrounding operation in place of the subexpression.

    Note that the pattern can therefore not be a single sequence variable/wildcard, because only single expressions
    will be matched.

    Args:
        expression:
            The expression to which the replacement rules are applied.
        rules:
            A collection of replacement rules that are applied to the expression.
        max_count:
            If given, at most *max_count* applications of the rules are performed. Otherwise, the rules
            are applied until there is no more match. If the set of replacement rules is not confluent,
            the replacement might not terminate without a *max_count* set.

    Returns:
        The resulting expression after the application of the replacement rules. This can also be a sequence of
        expressions, if the root expression is replaced with a sequence of expressions by a rule.
    """
    rules = [ReplacementRule(freeze(pattern), replacement) for pattern, replacement in rules]
    expression = freeze(expression)
    grouped = dict((h, list(g)) for h, g in itertools.groupby(rules, lambda r: r.pattern.head))
    replaced = True
    replace_count = 0
    while replaced and replace_count < max_count:
        replaced = False
        for subexpr, pos in expression.preorder_iter():
            if subexpr.head in grouped:
                for pattern, replacement in grouped[subexpr.head]:
                    try:
                        subst = next(match(subexpr, pattern))
                        result = replacement(**subst)
                        expression = freeze(replace(expression, pos, result))
                        replaced = True
                        break
                    except StopIteration:
                        pass
                if replaced:
                    break
        replace_count += 1

    return expression


def is_match(subject: Expression, pattern: Expression) -> bool:
    """
    Check whether the given subject expression matches given pattern.

    Args:
        subject:
            The subject for matching.
        pattern:
            The pattern for matching.

    Returns:
        True iff the subject matches the pattern.
    """
    return any(True for _ in match(subject, pattern))
