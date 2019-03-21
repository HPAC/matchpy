# -*- coding: utf-8 -*-
import pytest
from types import ModuleType

from matchpy.expressions.expressions import Pattern
from matchpy.matching.many_to_one import ManyToOneMatcher
from matchpy.matching.code_generation import CodeGenerator

from .test_matching import PARAM_MATCHES, PARAM_PATTERNS

GENERATED_TEMPLATE = '''
# -*- coding: utf-8 -*-
from matchpy import *
from tests.common import *
from tests.utils import *

{}

{}
'''.strip()


@pytest.mark.parametrize('subject, patterns', PARAM_PATTERNS.items())
def test_code_generation_many_to_one(subject, patterns):
    patterns = [Pattern(p) for p in patterns]
    matcher = ManyToOneMatcher(*patterns)

    generator = CodeGenerator(matcher)
    gc, code = generator.generate_code()
    code = GENERATED_TEMPLATE.format(gc, code)
    compiled = compile(code, '', 'exec')
    module = ModuleType('generated_code')
    print('=' * 80)
    print(code)
    print('=' * 80)
    exec(compiled, module.__dict__)

    for pattern in patterns:
        print(pattern)

    matches = list(module.match_root(subject))

    for i, pattern in enumerate(patterns):
        expected_matches = PARAM_MATCHES[subject, pattern.expression]
        for expected_match in expected_matches:
            assert (i, expected_match) in matches, "Subject {!s} and pattern {!s} did not yield the match {!s} but were supposed to".format(
                subject, pattern, expected_match
            )
            while (i, expected_match) in matches:
                matches.remove((i, expected_match))

    assert matches == [], "Subject {!s} and pattern {!s} yielded unexpected matches".format(
        subject, pattern
    )
