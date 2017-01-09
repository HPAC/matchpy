# -*- coding: utf-8 -*-
import random

import hypothesis.strategies as st
from hypothesis import assume, example, given
import pytest

from matchpy.expressions import (Arity, Atom, Operation, Symbol,
                                        Variable, Wildcard, freeze)
from matchpy.matching.one_to_one import match
from matchpy.matching.syntactic import OPERATION_END as OP_END
from matchpy.matching.syntactic import (DiscriminationNet, FlatTerm,
                                               SequenceMatcher, is_operation,
                                               is_symbol_wildcard)


class SpecialSymbol(Symbol): pass

f = Operation.new('f', Arity.variadic)
fc = Operation.new('fc', Arity.variadic, commutative=True)
g = Operation.new('g', Arity.variadic)
h = Operation.new('h', Arity.unary)
a = Symbol('a')
b = Symbol('b')
c = Symbol('c')
d = SpecialSymbol('d')
_ = Wildcard.dot()
x_ = Variable.dot('x')
__ = Wildcard.plus()
y__ = Variable.plus('y')
___ = Wildcard.star()
z___ = Variable.star('z')
_s = Wildcard.symbol(Symbol)


CONSTANT_EXPRESSIONS = [freeze(e) for e in [a, b, c, d]]


@pytest.mark.parametrize(
    '   expr,                   result',
    [
        (a,                     [a]),
        (_,                     [_]),
        (x_,                    [_]),
        (_s,                    [Symbol]),
        (Variable('v', f(_)),   [f, _, OP_END]),
        (f(),                   [f, OP_END]),
        (f(a),                  [f, a, OP_END]),
        (g(b),                  [g, b, OP_END]),
        (f(a, b),               [f, a, b, OP_END]),
        (f(x_),                 [f, _, OP_END]),
        (f(__),                 [f, __, OP_END]),
        (f(g(a)),               [f, g, a, OP_END, OP_END]),
        (f(g(a), b),            [f, g, a, OP_END, b, OP_END]),
        (f(a, g(b)),            [f, a, g, b, OP_END, OP_END]),
        (f(a, g(b), c),         [f, a, g, b, OP_END, c, OP_END]),
        (f(g(b), g(c)),         [f, g, b, OP_END, g, c, OP_END, OP_END]),
        (f(f(g(b)), g(c)),      [f, f, g, b, OP_END, OP_END, g, c, OP_END, OP_END]),
        (f(_, _),               [f, Wildcard.dot(2), OP_END]),
        (f(_, __),              [f, Wildcard(2, False), OP_END]),
        (f(_, __, __),          [f, Wildcard(3, False), OP_END]),
        (f(_, ___),             [f, __, OP_END]),
        (f(___, _),             [f, __, OP_END]),
        (f(___, __),            [f, __, OP_END]),
        (f(_, a, __),           [f, _, a, __, OP_END]),
    ]
)
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
    return st.lists(args, min_size=min_size, max_size=max_size).map(lambda a: freeze(func(*a)))


def expression_recurse_strategy(args):
    return func_wrap_strategy(args, f) | func_wrap_strategy(args, g)

expression_base_strategy = st.sampled_from([freeze(e) for e in [a, b, c, _, __, ___, _s]])
expression_strategy = st.recursive(expression_base_strategy, expression_recurse_strategy, max_leaves=10)


@pytest.mark.parametrize(
    '   pattern,                    expr,                                   is_match',
    [
        (a,                         a,                                      True),
        (_,                         a,                                      True),
        (_,                         g(a),                                   True),
        (_,                         g(h(a)),                                True),
        (_,                         g(a, b),                                True),
        (f(_),                      f(a),                                   True),
        (f(_),                      f(g(a)),                                True),
        (f(_),                      f(g(h(a))),                             True),
        (f(_),                      f(g(a, b)),                             True),
        (f(a, a),                   f(a),                                   False),
        (f(a, a),                   f(a, a),                                True),
        (f(a, a),                   f(a, b),                                False),
        (f(a, a),                   f(b, a),                                False),
        (f(__, a),                  f(a),                                   False),
        (f(__, a),                  f(a, a),                                True),
        (f(__, a),                  f(a, b, a),                             True),
        (f(__, a, a),               f(a, b, a, a),                          True),
        (f(__, a, a),               f(a, a, a),                             True),
        (f(__, a, a),               f(a),                                   False),
        (f(__, a, a),               f(a, a),                                False),
        (f(___, a),                 f(a),                                   True),
        (f(___, a),                 f(a, a),                                True),
        (f(___, a),                 f(a, b, a),                             True),
        (f(___, a, a),              f(a, b, a, a),                          True),
        (f(___, a, a),              f(a, a, a),                             True),
        (f(___, a, a),              f(a),                                   False),
        (f(___, a, a),              f(a, a),                                True),
        (f(___, a, b),              f(a, b, a, a),                          False),
        (f(___, a, b),              f(a, b, a, b),                          True),
        (f(___, a, b),              f(a, a, b),                             True),
        (f(___, a, b),              f(a, b, b),                             False),
        (f(___, a, b),              f(a, b),                                True),
        (f(___, a, b),              f(a, a),                                False),
        (f(___, a, b),              f(b, b),                                False),
        (f(___, a, _),              f(a),                                   False),
        (f(___, a, _),              f(),                                    False),
        (f(___, a, _),              f(b),                                   False),
        (f(___, a, _),              f(a, a),                                True),
        (f(___, a, _),              f(a, b),                                True),
        (f(___, a, _),              f(b, a),                                False),
        (f(___, a, _),              f(a, a, a),                             True),
        (f(___, a, _),              f(a, b, a),                             False),
        (f(___, a, _),              f(b, a, a),                             True),
        (f(___, a, _),              f(a, a, b),                             True),
        (f(___, a, _),              f(b, a, b),                             True),
        (f(___, a, _),              f(a, a, b, a, b),                       True),
        (f(___, a, _),              f(a, a, a, b, a, b),                    True),
        (f(___, a, __, a),          f(a, b, a),                             True),
        (f(___, a, __, a),          f(a, a, a),                             True),
        (f(___, a, __, a),          f(a, b, a, b),                          False),
        (f(___, a, __, a),          f(b, b, a, a),                          False),
        (f(___, a, __, a),          f(b, a, a, a),                          True),
        (f(___, a, __, a),          f(a, b, b, a),                          True),
        (f(___, g(a)),              f(g(a)),                                True),
        (f(___, g(a)),              f(a, g(a)),                             True),
        (f(___, g(a)),              f(g(a), g(a)),                          True),
        (f(___, g(a)),              f(g(a), g(b)),                          False),
        (f(___, g(a)),              f(g(b), g(a)),                          True),
        (f(___, g(_)),              f(g(a), g(b)),                          True),
        (f(___, g(_)),              f(g(b), g(a)),                          True),
        (f(___, g(_)),              f(g(b), g(h(a))),                       True),
        (f(___, g(_)),              f(g(b), g(g(a, b))),                    True),
        (f(___, g(_)),              f(g(b), g(h(a), a)),                    False),
        (f(___, g(___)),            f(g(a), g(b)),                          True),
        (f(___, g(___)),            f(g(b), g(a, b)),                       True),
        (f(___, g(___, a)),         f(g(a), g(b)),                          False),
        (f(___, g(___, a)),         f(g(b), g(a, b)),                       False),
        (f(___, g(___, a)),         f(g(b), g(a, a)),                       True),
        (f(___, g(___, a)),         f(g(a, b), g(a, a)),                    True),
        (f(___, g(___, a)),         f(g(b, b), g(b, a)),                    True),
        (f(___, g(___, a), b),      f(g(b, a), b, g(b, a)),                 False),
        (f(___, g(___, a), b),      f(g(b, a), b, g(b, a), b),              True),
        (f(___, g(h(a))),           f(g(a)),                                False),
        (f(___, g(h(a))),           f(g(h(b))),                             False),
        (f(___, g(h(a))),           f(g(h(a), b)),                          False),
        (f(___, g(h(a))),           f(g(h(a))),                             True),
        (f(___, a, a, b, ___),      f(a, a, a, b),                          True),
        (f(___, a, a, b),           f(a, a, a, b),                          True),
        (f(___, a, b),              f(a, a, a, b),                          True),
        (f(___, g(a), ___, g(b)),   f(g(b), g(a), g(a), g(b)),              True),
        (f(___, g(a), ___, g(b)),   f(g(a), g(a), g(b), g(b)),              True),
        (f(___, a, _s),             f(a, a),                                True),
        (f(___, a, _s),             f(a, a, a),                             True),
        (__,                        a,                                      True),
    ]
)
def test_generate_net_and_match(pattern, expr, is_match):
    net = DiscriminationNet()
    final_label = random.randrange(1000)
    net.add(freeze(pattern), final_label)
    result = net.match(freeze(expr))

    if is_match:
        assert result == [final_label], "Matching failed for {!s} and {!s}".format(pattern, expr)
    else:
        assert result == [], "Matching should fail for {!s} and {!s}".format(pattern, expr)


def test_variable_expression_match_error():
    net = DiscriminationNet()
    pattern = freeze(f(x_))
    net.add(pattern)

    with pytest.raises(TypeError):
        net.match(pattern)


@given(st.sets(expression_strategy, max_size=20))
@example({freeze(f(a)), freeze(f(_s))})
def test_randomized_product_net(patterns):
    assume(all(not isinstance(p, Atom) for p in patterns))

    patterns = list(patterns)
    net = DiscriminationNet()
    exprs = []
    for pattern in patterns:
        net.add(pattern)

        flatterm = []
        for term in FlatTerm(pattern):
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

    for pattern, expr in zip(patterns, exprs):
        result = net.match(expr)

        assert pattern in result, "{!s} did not match {!s} in the automaton".format(pattern, expr)

PRODUCT_NET_PATTERNS = [
    freeze(f(a, _, _)),
    freeze(f(_, a, _)),
    freeze(f(_, _, a)),
    freeze(f(__)),
    freeze(f(g(_), ___)),
    freeze(f(___, g(_))),
    freeze(_)
]

PRODUCT_NET_EXPRESSIONS = [
    f(a, a, a),
    f(b, a, a),
    f(a, b, a),
    f(a, a, b),
    f(g(a), a, a),
    f(g(a), a, g(b)),
    f(g(a), g(b), g(b)),
    f(a, g(b), g(b), a),
    f(g(a)),
]

@pytest.mark.xfail(reason="Currently this is broken in some cases, but it is unused, so not fixing it atm.")
@pytest.mark.parametrize('i', range(len(PRODUCT_NET_PATTERNS)))
def test_product_net(i):
    net = DiscriminationNet()

    patterns = PRODUCT_NET_PATTERNS[i:] + PRODUCT_NET_PATTERNS[:i]
    for pattern in patterns:
        net.add(pattern)

    for expression in PRODUCT_NET_EXPRESSIONS:
        result = net.match(expression)

        for pattern in patterns:
            try:
                next(match(expression, pattern))
            except StopIteration:
                assert pattern not in result, "Pattern {!s} should not match subject {!s}".format(pattern, expression)
            else:
                assert pattern in result, "Pattern {!s} should match subject {!s}".format(pattern, expression)


def test_sequence_matcher_match():
    PATTERNS = [
        f(___, x_, x_, ___),
        f(z___, a, b, ___),
        f(___, a, c, z___),
    ]

    matcher = SequenceMatcher(*PATTERNS)

    expr = freeze(f(a, b, c, a, a, b, a, c, b))

    matches = list(matcher.match(expr))

    assert len(matches) == 4
    assert (PATTERNS[0], {'x': a}) in matches
    assert (PATTERNS[1], {'z': ()}) in matches
    assert (PATTERNS[1], {'z': (a, b, c, a)}) in matches
    assert (PATTERNS[2], {'z': (b, )}) in matches

    assert list(matcher.match(freeze(a))) == []


@pytest.mark.parametrize(
    '   patterns,                   expected_error',
    [
        ([a],                       TypeError),
        ([fc(a)],                   TypeError),
        ([f(___, a, ___), g(a)],    TypeError),
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
)
def test_sequence_matcher_errors(patterns, expected_error):
    with pytest.raises(expected_error):
        SequenceMatcher(*patterns)
