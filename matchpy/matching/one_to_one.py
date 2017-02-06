# -*- coding: utf-8 -*-
from typing import Iterator, Tuple

from ..expressions.expressions import Expression
from ..expressions.substitution import Substitution
from .common import _match

__all__ = ['match', 'match_anywhere']


def match(subject: Expression, pattern: Expression) -> Iterator[Substitution]:
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
    return _match([subject], pattern, Substitution())


def match_anywhere(subject: Expression, pattern: Expression) -> Iterator[Tuple[Substitution, Tuple[int, ...]]]:
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
    if pattern.head is None:
        child_iterator = subject.preorder_iter()
    else:
        child_iterator = subject.preorder_iter(lambda child: child.head == pattern.head)
    for child, pos in child_iterator:
        for subst in _match([child], pattern, Substitution()):
            yield subst, pos
