# -*- coding: utf-8 -*-
import random

from hypothesis import assume, example, given
import hypothesis.strategies as st
import pytest

from matchpy.expressions.expressions import Atom, Operation, Symbol, Wildcard, Pattern
from matchpy.matching.one_to_one import match
from matchpy.matching.syntactic import OPERATION_END as OP_END
from matchpy.matching.syntactic import DiscriminationNet, FlatTerm, SequenceMatcher, is_operation, is_symbol_wildcard
from .common import *

CONSTANT_EXPRESSIONS = [e for e in [a, b, c, d]]


@pytest.mark.parametrize(
    '   expr,                   result',
    [
        (a,                     [a]),
        (_,                     [_]),
        (x_,                    [_]),
        (_s,                    [Symbol]),
        (f(_, variable_name='v'),    [f, _, OP_END]),
        (f(),                   [f, OP_END]),
        (f(a),                  [f, a, OP_END]),
        (f2(b),                 [f2, b, OP_END]),
        (f(a, b),               [f, a, b, OP_END]),
        (f(x_),                 [f, _, OP_END]),
        (f(__),                 [f, __, OP_END]),
        (f(f2(a)),              [f, f2, a, OP_END, OP_END]),
        (f(f2(a), b),           [f, f2, a, OP_END, b, OP_END]),
        (f(a, f2(b)),           [f, a, f2, b, OP_END, OP_END]),
        (f(a, f2(b), c),        [f, a, f2, b, OP_END, c, OP_END]),
        (f(f2(b), f2(c)),       [f, f2, b, OP_END, f2, c, OP_END, OP_END]),
        (f(f(f2(b)), f2(c)),    [f, f, f2, b, OP_END, OP_END, f2, c, OP_END, OP_END]),
        (f(_, _),               [f, Wildcard(2, True), OP_END]),
        (f(_, __),              [f, Wildcard(2, False), OP_END]),
        (f(_, __, __),          [f, Wildcard(3, False), OP_END]),
        (f(_, ___),             [f, __, OP_END]),
        (f(___, _),             [f, __, OP_END]),
        (f(___, __),            [f, __, OP_END]),
        (f(_, a, __),           [f, _, a, __, OP_END]),
    ]
)  # yapf: disable
def test_flatterm_init(expr, result):
    term = list(FlatTerm(expr))
    assert term == result


def test_flatterm_init_error():
    with pytest.raises(TypeError):
        FlatTerm(None)


def test_flatterm_add():
    assert FlatTerm(a) + [b] == FlatTerm([a, b])
    assert FlatTerm(a) + (b, ) == FlatTerm([a, b])
    assert FlatTerm(a) + FlatTerm(b) == FlatTerm([a, b])

    with pytest.raises(TypeError):
        FlatTerm() + 5


def test_flatterm_contains():
    flatterm = FlatTerm(f(a))

    assert f in flatterm
    assert a in flatterm
    assert b not in flatterm


def test_flatterm_getitem():
    flatterm = FlatTerm(f(a))

    assert flatterm[0] == f
    assert flatterm[1] == a
    assert flatterm[2] == OP_END

    with pytest.raises(IndexError):
        flatterm[3]


def test_flatterm_eq():
    assert FlatTerm(a) == FlatTerm(a)
    assert not FlatTerm(a) == FlatTerm(b)
    assert not FlatTerm(a) == [a]
    assert not FlatTerm(a) == FlatTerm(f(a))
    assert FlatTerm(f(a)) == FlatTerm(f(a))
    assert not FlatTerm(f(a)) == FlatTerm(f(b))


@pytest.mark.parametrize(
    '   flatterm,           is_syntactic',
    [
        (a,                 True),
        (f(a),              True),
        (_,                 True),
        (__,                False),
        (f_c(a),            False),
        (f(_, _),           True),
        (f(__),             False),
        (f(x_),             True),
        (f(z___),           False),
    ]
)  # yapf: disable
def test_flatterm_is_syntactic(flatterm, is_syntactic):
    flatterm = FlatTerm(flatterm)
    assert flatterm.is_syntactic == is_syntactic


def test_is_operation():
    assert is_operation(str) is False
    assert is_operation(1) is False
    assert is_operation(None) is False
    assert is_operation(Operation) is True
    assert is_operation(f) is True


def test_is_symbol_wildcard():
    assert is_symbol_wildcard(str) is False
    assert is_symbol_wildcard(1) is False
    assert is_symbol_wildcard(None) is False
    assert is_symbol_wildcard(Symbol) is True
    assert is_symbol_wildcard(SpecialSymbol) is True


def func_wrap_strategy(args, func):
    min_size = func.arity[0]
    max_size = func.arity[1] and func.arity[0] or 4
    return st.lists(args, min_size=min_size, max_size=max_size).map(lambda a: func(*a))


def expression_recurse_strategy(args):
    return func_wrap_strategy(args, f) | func_wrap_strategy(args, f2)


expression_base_strategy = st.sampled_from([e for e in [a, b, c, _, __, ___, _s]])
expression_strategy = st.recursive(expression_base_strategy, expression_recurse_strategy, max_leaves=10)


@pytest.mark.parametrize(
    '   pattern,                    expr,                                is_match',
    [
        (a,                         a,                                   True),
        (_,                         a,                                   True),
        (_,                         f2(a),                               True),
        (_,                         f2(f_u(a)),                          True),
        (_,                         f2(a, b),                            True),
        (f(_),                      f(a),                                True),
        (f(_),                      f(f2(a)),                            True),
        (f(_),                      f(f2(f_u(a))),                       True),
        (f(_),                      f(f2(a, b)),                         True),
        (f(a, a),                   f(a),                                False),
        (f(a, a),                   f(a, a),                             True),
        (f(a, a),                   f(a, b),                             False),
        (f(a, a),                   f(b, a),                             False),
        (f(__, a),                  f(a),                                False),
        (f(__, a),                  f(a, a),                             True),
        (f(__, a),                  f(a, b, a),                          True),
        (f(__, a, a),               f(a, b, a, a),                       True),
        (f(__, a, a),               f(a, a, a),                          True),
        (f(__, a, a),               f(a),                                False),
        (f(__, a, a),               f(a, a),                             False),
        (f(___, a),                 f(a),                                True),
        (f(___, a),                 f(a, a),                             True),
        (f(___, a),                 f(a, b, a),                          True),
        (f(___, a, a),              f(a, b, a, a),                       True),
        (f(___, a, a),              f(a, a, a),                          True),
        (f(___, a, a),              f(a),                                False),
        (f(___, a, a),              f(a, a),                             True),
        (f(___, a, b),              f(a, b, a, a),                       False),
        (f(___, a, b),              f(a, b, a, b),                       True),
        (f(___, a, b),              f(a, a, b),                          True),
        (f(___, a, b),              f(a, b, b),                          False),
        (f(___, a, b),              f(a, b),                             True),
        (f(___, a, b),              f(a, a),                             False),
        (f(___, a, b),              f(b, b),                             False),
        (f(___, a, _),              f(a),                                False),
        (f(___, a, _),              f(),                                 False),
        (f(___, a, _),              f(b),                                False),
        (f(___, a, _),              f(a, a),                             True),
        (f(___, a, _),              f(a, b),                             True),
        (f(___, a, _),              f(b, a),                             False),
        (f(___, a, _),              f(a, a, a),                          True),
        (f(___, a, _),              f(a, b, a),                          False),
        (f(___, a, _),              f(b, a, a),                          True),
        (f(___, a, _),              f(a, a, b),                          True),
        (f(___, a, _),              f(b, a, b),                          True),
        (f(___, a, _),              f(a, a, b, a, b),                    True),
        (f(___, a, _),              f(a, a, a, b, a, b),                 True),
        (f(___, a, __, a),          f(a, b, a),                          True),
        (f(___, a, __, a),          f(a, a, a),                          True),
        (f(___, a, __, a),          f(a, b, a, b),                       False),
        (f(___, a, __, a),          f(b, b, a, a),                       False),
        (f(___, a, __, a),          f(b, a, a, a),                       True),
        (f(___, a, __, a),          f(a, b, b, a),                       True),
        (f(___, f2(a)),             f(f2(a)),                            True),
        (f(___, f2(a)),             f(a, f2(a)),                         True),
        (f(___, f2(a)),             f(f2(a), f2(a)),                     True),
        (f(___, f2(a)),             f(f2(a), f2(b)),                     False),
        (f(___, f2(a)),             f(f2(b), f2(a)),                     True),
        (f(___, f2(_)),             f(f2(a), f2(b)),                     True),
        (f(___, f2(_)),             f(f2(b), f2(a)),                     True),
        (f(___, f2(_)),             f(f2(b), f2(f_u(a))),                True),
        (f(___, f2(_)),             f(f2(b), f2(f2(a, b))),              True),
        (f(___, f2(_)),             f(f2(b), f2(f_u(a), a)),             False),
        (f(___, f2(___)),           f(f2(a), f2(b)),                     True),
        (f(___, f2(___)),           f(f2(b), f2(a, b)),                  True),
        (f(___, f2(___, a)),        f(f2(a), f2(b)),                     False),
        (f(___, f2(___, a)),        f(f2(b), f2(a, b)),                  False),
        (f(___, f2(___, a)),        f(f2(b), f2(a, a)),                  True),
        (f(___, f2(___, a)),        f(f2(a, b), f2(a, a)),               True),
        (f(___, f2(___, a)),        f(f2(b, b), f2(b, a)),               True),
        (f(___, f2(___, a), b),     f(f2(b, a), b, f2(b, a)),            False),
        (f(___, f2(___, a), b),     f(f2(b, a), b, f2(b, a), b),         True),
        (f(___, f2(f_u(a))),        f(f2(a)),                            False),
        (f(___, f2(f_u(a))),        f(f2(f_u(b))),                       False),
        (f(___, f2(f_u(a))),        f(f2(f_u(a), b)),                    False),
        (f(___, f2(f_u(a))),        f(f2(f_u(a))),                       True),
        (f(___, a, a, b, ___),      f(a, a, a, b),                       True),
        (f(___, a, a, b),           f(a, a, a, b),                       True),
        (f(___, a, b),              f(a, a, a, b),                       True),
        (f(___, f2(a), ___, f2(b)), f(f2(b), f2(a), f2(a), f2(b)),       True),
        (f(___, f2(a), ___, f2(b)), f(f2(a), f2(a), f2(b), f2(b)),       True),
        (f(___, a, _s),             f(a, a),                             True),
        (f(___, a, _s),             f(a, a, a),                          True),
        (__,                        a,                                   True),
    ]
)  # yapf: disable
def test_generate_net_and_match(pattern, expr, is_match):
    net = DiscriminationNet()
    index = net.add(Pattern(pattern))
    result = net._match(expr)

    if is_match:
        assert result == [index], "Matching failed for {!s} and {!s}".format(pattern, expr)
    else:
        assert result == [], "Matching should fail for {!s} and {!s}".format(pattern, expr)


def test_variable_expression_match_error():
    net = DiscriminationNet()
    pattern = Pattern(f(x_))
    net.add(pattern)

    with pytest.raises(TypeError):
        list(net.match(pattern))


@given(st.sets(expression_strategy, max_size=20))
@example({f(a), f(_s)})
def test_randomized_product_net(patterns):
    assume(all(not isinstance(p, Atom) for p in patterns))

    patterns = [Pattern(p) for p in patterns]
    net = DiscriminationNet()
    exprs = []
    for pattern in patterns:
        net.add(pattern)

        flatterm = []
        for term in FlatTerm(pattern.expression):
            if isinstance(term, Wildcard):
                args = [random.choice(CONSTANT_EXPRESSIONS) for _ in range(term.min_count)]
                flatterm.extend(args)
            elif is_symbol_wildcard(term):
                flatterm.append(random.choice(CONSTANT_EXPRESSIONS))
            else:
                flatterm.append(term)

        if not flatterm:
            flatterm = [random.choice(CONSTANT_EXPRESSIONS)]
        exprs.append(flatterm)

    for index, (pattern, expr) in enumerate(zip(patterns, exprs)):
        result = net._match(expr)
        assert index in result, "{!s} did not match {!s} in the DiscriminationNet".format(pattern, expr)


PRODUCT_NET_PATTERNS = [
    Pattern(f(a, _, _)),
    Pattern(f(_, a, _)),
    Pattern(f(_, _, a)),
    Pattern(f(__)),
    Pattern(f(f2(_, ___))),
    Pattern(f(___, f2(_))),
    Pattern(_),
]

PRODUCT_NET_EXPRESSIONS = [
    f(a, a, a),
    f(b, a, a),
    f(a, b, a),
    f(a, a, b),
    f(f2(a), a, a),
    f(f2(a), a, f2(b)),
    f(f2(a), f2(b), f2(b)),
    f(a, f2(b), f2(b), a),
    f(f2(a)),
]


@pytest.mark.xfail(reason="Currently this is broken in some cases, but it is unused, so not fixing it atm.")
@pytest.mark.parametrize('i', range(len(PRODUCT_NET_PATTERNS)))
def test_product_net(i):
    net = DiscriminationNet()

    patterns = PRODUCT_NET_PATTERNS[i:] + PRODUCT_NET_PATTERNS[:i]
    for pattern in patterns:
        net.add(pattern)

    for expression in PRODUCT_NET_EXPRESSIONS:
        result = [p for p, _ in net.match(expression)]

        for pattern in patterns:
            try:
                next(match(expression, pattern))
            except StopIteration:
                assert pattern not in result, "Pattern {!s} should not match subject {!s}".format(pattern, expression)
            else:
                assert pattern in result, "Pattern {!s} should match subject {!s}".format(pattern, expression)


def test_sequence_matcher_match():
    PATTERNS = [
        Pattern(f(___, x_, x_, ___)),
        Pattern(f(z___, a, b, ___)),
        Pattern(f(___, a, c, z___)),
        Pattern(f(z___, a, c, z___)),
    ]

    matcher = SequenceMatcher(*PATTERNS)

    expr = f(a, b, c, a, a, b, a, c, b)

    matches = list(matcher.match(expr))

    assert len(matches) == 4
    assert (PATTERNS[0], {'x': a}) in matches
    assert (PATTERNS[1], {'z': ()}) in matches
    assert (PATTERNS[1], {'z': (a, b, c, a)}) in matches
    assert (PATTERNS[2], {'z': (b, )}) in matches

    assert list(matcher.match(a)) == []


@pytest.mark.parametrize(
    '   patterns,                   expected_error',
    [
        ([a],                       TypeError),
        ([f_c(a)],                  TypeError),
        ([f(___, a, ___), f2(a)],   TypeError),
        ([f(___)],                  ValueError),
        ([f(___, a)],               ValueError),
        ([f(a, b, c)],              ValueError),
        ([f(_, b, ___)],            ValueError),
        ([f(___, b, _)],            ValueError),
        ([f(__, b, ___)],           ValueError),
        ([f(___, b, __)],           ValueError),
        ([f(a, b, ___)],            ValueError),
        ([f(___, b, c)],            ValueError),
    ]
)  # yapf: disable
def test_sequence_matcher_errors(patterns, expected_error):
    with pytest.raises(expected_error):
        SequenceMatcher(*map(Pattern, patterns))


@pytest.mark.parametrize(
    '   pattern,                    can_match',
    [
        (a,                         False),
        (f_c(a),                    False),
        (f(___),                    False),
        (f(___, a),                 False),
        (f(a, b, c),                False),
        (f(_, b, ___),              False),
        (f(___, b, _),              False),
        (f(__, b, ___),             False),
        (f(___, b, __),             False),
        (f(a, b, ___),              False),
        (f(___, b, c),              False),
        (f(___, b, c),              False),
        (f(___, b, c, ___),         True),
        (f(___, b, ___),            True),
        (f(___, f2(x_), ___),       True),
    ]
)  # yapf: disable
def test_sequence_matcher_can_match(pattern, can_match):
    assert SequenceMatcher.can_match(Pattern(pattern)) == can_match
