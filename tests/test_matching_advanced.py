# -*- coding: utf-8 -*-
from hypothesis import assume, given
import hypothesis.strategies as st
import pytest
from multiset import Multiset

from matchpy.expressions import Symbol, Wildcard, Pattern, Alternatives, ExpressionSequence, Repeated
from .common import *
from matchpy.matching.one_to_one import match


@pytest.mark.parametrize(
    '   subject,            pattern,                is_match',
    [
        (a,                 Alternatives(a, b),     True),
        (b,                 Alternatives(a, b),     True),
        (c,                 Alternatives(a, b),     False),
    ]
)  # yapf: disable
def test_alternatives_match(subject, pattern, is_match):
    subject = subject
    pattern = Pattern(pattern)
    result = list(match(subject, pattern))
    if is_match:
        assert result == [dict()], "Subject {!s} and pattern {!s} did not match but were supposed to".format(
            subject, pattern
        )
    else:
        assert result == [], "Subject {!s} and pattern {!s} did match but were not supposed to".format(
            subject, pattern
        )

@pytest.mark.parametrize(
    '   subject,        pattern,                                        is_match',
    [
        (f(a, a),       f(Alternatives(ExpressionSequence(a, a), b)),   True),
        (f(a, b),       f(Alternatives(ExpressionSequence(a, a), b)),   False),
        (f(b, a),       f(Alternatives(ExpressionSequence(a, a), b)),   False),
        (f(b, b),       f(Alternatives(ExpressionSequence(a, a), b)),   False),
        (f(a),          f(Alternatives(ExpressionSequence(a, a), b)),   False),
        (f(b),          f(Alternatives(ExpressionSequence(a, a), b)),   True),
    ]
)  # yapf: disable
def test_alternatives_with_sequence_match(subject, pattern, is_match):
    subject = subject
    pattern = Pattern(pattern)
    result = list(match(subject, pattern))
    if is_match:
        assert result == [dict()], "Subject {!s} and pattern {!s} did not match but were supposed to".format(
            subject, pattern
        )
    else:
        assert result == [], "Subject {!s} and pattern {!s} did match but were not supposed to".format(
            subject, pattern
        )

@pytest.mark.parametrize(
    '   subject,        pattern,           is_match',
    [
        (f(a),          f(Repeated(a)),   True),
        (f(a, a),       f(Repeated(a)),   True),
        (f(a, a, a),    f(Repeated(a)),   True),
        (f(b),          f(Repeated(a)),   False),
        (f(a, b),       f(Repeated(a)),   False),
        (f(a, b, a),    f(Repeated(a)),   False),
        (f(b, a),       f(Repeated(a)),   False),
        (f(b, b),       f(Repeated(a)),   False),
        (f(b, b),       f(Repeated(x_)),  True),
        (f(a, a),       f(Repeated(x_)),  True),
        (f(a, b),       f(Repeated(x_)),  False),
        (f(a),          f(Repeated(x_)),  True),
        (f(b),          f(Repeated(x_)),  True),
        (f(a, a, b),    f(Repeated(x_)),  False),
    ]
)  # yapf: disable
def test_repeated_match(subject, pattern, is_match):
    pattern = Pattern(pattern)
    result = list(match(subject, pattern))
    if is_match:
        assert len(result) > 0, "Subject {!s} and pattern {!s} did not match but were supposed to".format(
            subject, pattern
        )
    else:
        assert len(result) == 0, "Subject {!s} and pattern {!s} did match but were not supposed to".format(
            subject, pattern
        )
