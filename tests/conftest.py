# -*- coding: utf-8 -*-
import pytest
from types import ModuleType

from matchpy.expressions.expressions import Wildcard, CommutativeOperation
from matchpy.matching.one_to_one import match as match_one_to_one
from matchpy.matching.many_to_one import ManyToOneMatcher
from matchpy.matching.syntactic import DiscriminationNet
from matchpy.expressions.functions import preorder_iter
from matchpy.matching.code_generation import CodeGenerator

def pytest_configure():
    pytest.matcher = None

def pytest_generate_tests(metafunc):
    if 'match' in metafunc.fixturenames:
        metafunc.parametrize('match', ['one-to-one', 'many-to-one', 'generated'], indirect=True)
    if 'match_syntactic' in metafunc.fixturenames:
        metafunc.parametrize('match_syntactic', ['one-to-one', 'many-to-one', 'syntactic', 'generated'], indirect=True)
    if 'match_many' in metafunc.fixturenames:
        metafunc.parametrize('match_many', ['many-to-one', 'generated'], indirect=True)


def match_many_to_one(expression, *patterns):
    try:
        pattern = patterns[0]
        commutative = next(
            p for p in preorder_iter(pattern.expression) if isinstance(p, CommutativeOperation)
        )
        next(wc for wc in preorder_iter(commutative) if isinstance(wc, Wildcard) and wc.min_count > 1)
    except StopIteration:
        pass
    else:
        pytest.xfail('Matcher does not support fixed wildcards with length != 1 in commutative operations')
    matcher = ManyToOneMatcher(*patterns)
    for _, substitution in matcher.match(expression):
        yield substitution


GENERATED_TEMPLATE = '''
# -*- coding: utf-8 -*-
from matchpy import *
from tests.common import *
from tests.utils import *

{}

{}
'''.strip()


def match_generated(expression, *patterns):
    matcher = ManyToOneMatcher(*patterns)
    generator = CodeGenerator(matcher)
    gc, code = generator.generate_code()
    code = GENERATED_TEMPLATE.format(gc, code)
    compiled = compile(code, '', 'exec')
    module = ModuleType("generated_code")
    print(code)
    exec(compiled, module.__dict__)
    for _, substitution in module.match_root(expression):
        yield substitution



def syntactic_matcher(expression, pattern):
    matcher = DiscriminationNet()
    matcher.add(pattern)
    for _, substitution in matcher.match(expression):
        yield substitution


@pytest.fixture
def match(request):
    pytest.matcher = request.param
    if request.param == 'one-to-one':
        return match_one_to_one
    elif request.param == 'many-to-one':
        return match_many_to_one
    elif request.param == 'generated':
        return match_generated
    else:
        raise ValueError("Invalid internal test config")


@pytest.fixture
def match_syntactic(request):
    pytest.matcher = request.param
    if request.param == 'one-to-one':
        return match_one_to_one
    elif request.param == 'many-to-one':
        return match_many_to_one
    elif request.param == 'syntactic':
        return syntactic_matcher
    elif request.param == 'generated':
        return match_generated
    else:
        raise ValueError("Invalid internal test config")


@pytest.fixture
def match_many(request):
    pytest.matcher = request.param
    if request.param == 'many-to-one':
        return match_many_to_one
    elif request.param == 'generated':
        return match_generated
    else:
        raise ValueError("Invalid internal test config")
