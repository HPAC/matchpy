# -*- coding: utf-8 -*-
import pytest

from matchpy.expressions.expressions import Operation, Wildcard
from matchpy.matching.one_to_one import match as match_one_to_one
from matchpy.matching.many_to_one import ManyToOneMatcher
from matchpy.matching.syntactic import DiscriminationNet


def pytest_generate_tests(metafunc):
    if 'match' in metafunc.fixturenames:
        metafunc.parametrize('match', ['one-to-one', 'many-to-one'], indirect=True)
    if 'match_syntactic' in metafunc.fixturenames:
        metafunc.parametrize('match_syntactic', ['one-to-one', 'many-to-one', 'syntactic'], indirect=True)


def match_many_to_one(expression, pattern):
    try:
        commutative, _ = next(
            p for p in pattern.expression.preorder_iter(lambda e: isinstance(e, Operation) and e.commutative)
        )
        next(wc for wc in commutative.preorder_iter(lambda e: isinstance(e, Wildcard) and e.min_count > 1))
    except StopIteration:
        pass
    else:
        pytest.xfail('Matcher does not support fixed wildcards with length != 1 in commutative operations')
    matcher = ManyToOneMatcher(pattern)
    for _, substitution in matcher.match(expression):
        yield substitution


def syntactic_matcher(expression, pattern):
    matcher = DiscriminationNet()
    matcher.add(pattern)
    for _, substitution in matcher.match(expression):
        yield substitution


@pytest.fixture
def match(request):
    if request.param == 'one-to-one':
        return match_one_to_one
    elif request.param == 'many-to-one':
        return match_many_to_one
    else:
        raise ValueError("Invalid internal test config")


@pytest.fixture
def match_syntactic(request):
    if request.param == 'one-to-one':
        return match_one_to_one
    elif request.param == 'many-to-one':
        return match_many_to_one
    elif request.param == 'syntactic':
        return syntactic_matcher
    else:
        raise ValueError("Invalid internal test config")
