# -*- coding: utf-8 -*-
import pytest

from patternmatcher.expressions import Operation, Symbol, Variable, Arity, Wildcard, freeze
from patternmatcher.matching.many_to_one import ManyToOneMatcher
from patternmatcher.matching.one_to_one import match as match_one_to_one
from patternmatcher.matching.automaton import Automaton

@pytest.fixture(autouse=True)
def add_default_expressions(doctest_namespace):
    doctest_namespace['f'] = Operation.new('f', Arity.variadic)
    doctest_namespace['a'] = Symbol('a')
    doctest_namespace['b'] = Symbol('b')
    doctest_namespace['c'] = Symbol('c')
    doctest_namespace['x_'] = Variable.dot('x')
    doctest_namespace['y_'] = Variable.dot('y')
    doctest_namespace['_'] = Wildcard.dot()
    doctest_namespace['__'] = Wildcard.plus()
    doctest_namespace['___'] = Wildcard.star()
    doctest_namespace['freeze'] = freeze
    doctest_namespace['__name__'] = '__main__'

def pytest_generate_tests(metafunc):
    if 'match' in metafunc.fixturenames:
        metafunc.parametrize('match', ['one-to-one', 'many-to-one', 'automaton'], indirect=True)


def match_many_to_one(expression, pattern):
    matcher = ManyToOneMatcher(pattern)
    for _, substitution in matcher.match(expression):
        yield substitution

def match_automaton(expression, pattern):
    try:
        next(p for p in pattern.preorder_iter(lambda e: isinstance(e, Operation) and e.commutative))
    except StopIteration:
        pass
    else:
        pytest.xfail('Matcher does not support commutative operations (yet)')
    matcher = Automaton()
    matcher.add(pattern)
    for _, substitution in matcher.match(expression):
        yield substitution

match_automaton.xfail = True

@pytest.fixture
def match(request):
    if request.param == 'one-to-one':
        return match_one_to_one
    elif request.param == 'many-to-one':
        return match_many_to_one
    elif request.param == 'automaton':
        return match_automaton
    else:
        raise ValueError("Invalid internal test config")
