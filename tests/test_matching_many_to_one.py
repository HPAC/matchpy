# -*- coding: utf-8 -*-
import pytest

from matchpy.expressions.constraints import CustomConstraint
from matchpy.expressions.expressions import Symbol, Pattern
from matchpy.matching.many_to_one import ManyToOneMatcher
from .common import *
from .utils import MockConstraint


def test_add_duplicate_pattern():
    pattern = Pattern(f(a))
    matcher = ManyToOneMatcher()

    matcher.add(pattern)
    matcher.add(pattern)

    assert len(matcher.patterns) == 1


def test_add_duplicate_pattern_with_different_constraint():
    pattern1 = Pattern(f(a))
    pattern2 = Pattern(f(a), MockConstraint(False))
    matcher = ManyToOneMatcher()

    matcher.add(pattern1)
    matcher.add(pattern2)

    assert len(matcher.patterns) == 2


def test_different_constraints():
    c1 = CustomConstraint(lambda x: len(str(x)) > 1)
    c2 = CustomConstraint(lambda x: len(str(x)) == 1)
    pattern1 = Pattern(f(x_), c1)
    pattern2 = Pattern(f(x_), c2)
    pattern3 = Pattern(f(x_, b), c1)
    pattern4 = Pattern(f(x_, b), c2)
    matcher = ManyToOneMatcher(pattern1, pattern2, pattern3, pattern4)

    subject = f(a)
    results = list(matcher.match(subject))
    assert len(results) == 1
    assert results[0][0] == pattern2
    assert results[0][1] == {'x': a}

    subject = f(Symbol('longer'), b)
    results = sorted(matcher.match(subject))
    assert len(results) == 1
    assert results[0][0] == pattern3
    assert results[0][1] == {'x': Symbol('longer')}


def test_different_constraints_with_match_on_operation():
    c1 = CustomConstraint(lambda x: len(str(x)) > 1)
    c2 = CustomConstraint(lambda x: len(str(x)) == 1)
    pattern1 = Pattern(f(x_), c1)
    pattern2 = Pattern(f(x_), c2)
    pattern3 = Pattern(f(x_, b), c1)
    pattern4 = Pattern(f(x_, b), c2)
    matcher = ManyToOneMatcher(pattern1, pattern2, pattern3, pattern4)

    subject = f(a)
    results = list(matcher.match(subject))
    assert len(results) == 1
    assert results[0][0] == pattern2
    assert results[0][1] == {'x': a}

    subject = f(Symbol('longer'), b)
    results = sorted(matcher.match(subject))
    assert len(results) == 1
    assert results[0][0] == pattern3
    assert results[0][1] == {'x': Symbol('longer')}


def test_different_constraints_no_match_on_operation():
    c1 = CustomConstraint(lambda x: x == a)
    c2 = CustomConstraint(lambda x: x == b)
    pattern1 = Pattern(f(x_), c1)
    pattern2 = Pattern(f(x_), c2)
    matcher = ManyToOneMatcher(pattern1, pattern2)

    subject = f(c)
    results = list(matcher.match(subject))
    assert len(results) == 0


def test_different_constraints_on_commutative_operation():
    c1 = CustomConstraint(lambda x: len(str(x)) > 1)
    c2 = CustomConstraint(lambda x: len(str(x)) == 1)
    pattern1 = Pattern(f_c(x_), c1)
    pattern2 = Pattern(f_c(x_), c2)
    pattern3 = Pattern(f_c(x_, b), c1)
    pattern4 = Pattern(f_c(x_, b), c2)
    matcher = ManyToOneMatcher(pattern1, pattern2, pattern3, pattern4)

    subject = f_c(a)
    results = list(matcher.match(subject))
    assert len(results) == 1
    assert results[0][0] == pattern2
    assert results[0][1] == {'x': a}

    subject = f_c(Symbol('longer'), b)
    results = sorted(matcher.match(subject))
    assert len(results) == 1
    assert results[0][0] == pattern3
    assert results[0][1] == {'x': Symbol('longer')}

    subject = f_c(a, b)
    results = list(matcher.match(subject))
    assert len(results) == 1
    assert results[0][0] == pattern4
    assert results[0][1] == {'x': a}


@pytest.mark.parametrize('c1', [True, False])
@pytest.mark.parametrize('c2', [True, False])
def test_different_pattern_same_constraint(c1, c2):
    constr1 = CustomConstraint(lambda x: c1)
    constr2 = CustomConstraint(lambda x: c2)
    constr3 = CustomConstraint(lambda x: True)
    patterns = [
        Pattern(f2(x_, a), constr3),
        Pattern(f(a, a, x_), constr3),
        Pattern(f(a, x_), constr1),
        Pattern(f(x_, a), constr2),
        Pattern(f(a, x_, b), constr1),
        Pattern(f(x_, a, b), constr1),
    ]
    subject = f(a, a)

    matcher = ManyToOneMatcher(*patterns)
    results = list(matcher.match(subject))

    assert len(results) == int(c1) + int(c2)


def test_same_commutative_but_different_pattern():
    pattern1 = Pattern(f(f_c(x_), a))
    pattern2 = Pattern(f(f_c(x_), b))
    matcher = ManyToOneMatcher(pattern1, pattern2)

    subject = f(f_c(a), a)
    result = list(matcher.match(subject))
    assert result == [(pattern1, {'x': a})]

    subject = f(f_c(a), b)
    result = list(matcher.match(subject))
    assert result == [(pattern2, {'x': a})]


def test_grouped():
    pattern1 = Pattern(a, MockConstraint(True))
    pattern2 = Pattern(a, MockConstraint(True))
    pattern3 = Pattern(x_, MockConstraint(True))
    matcher = ManyToOneMatcher(pattern1, pattern2, pattern3)

    result = [[p for p, _ in ps] for ps in matcher.match(a).grouped()]

    assert len(result) == 2
    for res in result:
        if len(res) == 2:
            assert pattern1 in res
            assert pattern2 in res
        elif len(res) == 1:
            assert pattern3 in res
        else:
            assert False, "Wrong number of grouped matches"


def test_same_pattern_different_label():
    pattern = Pattern(a)
    matcher = ManyToOneMatcher()
    matcher.add(pattern, 42)
    matcher.add(pattern, 23)

    result = sorted((l, sorted(map(tuple, s.items()))) for l, s in matcher.match(a))

    assert result == [(23, []), (42, [])]


def test_different_pattern_same_label():
    matcher = ManyToOneMatcher()
    matcher.add(Pattern(a), 42)
    matcher.add(Pattern(x_), 42)

    result = sorted((l, sorted(map(tuple, s.items()))) for l, s in matcher.match(a))

    assert result == [(42, []), (42, [('x', a)])]


def test_different_pattern_different_label():
    matcher = ManyToOneMatcher()
    matcher.add(Pattern(a), 42)
    matcher.add(Pattern(x_), 23)

    result = sorted((l, sorted(map(tuple, s.items()))) for l, s in matcher.match(a))

    assert result == [(23, [('x', a)]), (42, [])]
