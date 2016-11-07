# -*- coding: utf-8 -*-
import pytest
import random

import hypothesis.strategies as st
from hypothesis import given

from patternmatcher.expressions import (Arity, Operation, Symbol, Variable,
                                        Wildcard, freeze)
from patternmatcher.syntactic import OPERATION_END as OP_END
from patternmatcher.syntactic import DiscriminationNet, FlatTerm, is_operation, is_symbol_wildcard

class SpecialSymbol(Symbol): pass

f = Operation.new('f', Arity.variadic)
g = Operation.new('g', Arity.variadic)
h = Operation.new('h', Arity.variadic)
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


@pytest.mark.parametrize('expr,result', [
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
])
def test_flatterm_init(expr, result):
    term = list(FlatTerm(expr))
    assert term == result


def test_flatterm_init_error():
    with pytest.raises(TypeError):
        FlatTerm(None)


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


@pytest.mark.parametrize('pattern,expr,is_match', [
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
   # (f(___, a, _),              f(a, b, a),                             False),  # TODO: currently failing
    (f(___, a, _),              f(b, a, a),                             True),
  #  (f(___, a, _),              f(a, a, b),                             True),   # TODO: currently failing
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
    (f(___, g(_)),              f(g(b), g(h(a, b))),                    True),
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
])
def test_generate_net_and_match(pattern, expr, is_match):
    net = DiscriminationNet()
    net._net = DiscriminationNet._generate_net(freeze(pattern))
    result = net.match(freeze(expr))

    if is_match:
        assert result == [pattern], 'Matching failed for %s and %s' % (pattern, expr)
    else:
        assert result == [], 'Matching should fail for %s and %s' % (pattern, expr)


def test_variable_expression_match_error():
    net = DiscriminationNet()
    pattern = freeze(f(x_))
    net.add(pattern)

    with pytest.raises(TypeError):
        net.match(pattern)


@pytest.mark.skip('Takes too long')
@given(st.sets(expression_strategy, max_size=20))
def test_randomized_product_net(patterns):
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

    net_id = hash(frozenset(patterns))

    for pattern, expr in zip(patterns, exprs):
        result = net.match(expr)

        assert pattern in result, '%s %s' % (expr, net_id)


if __name__ == '__main__':
    import patternmatcher.syntactic as tested_module
    pytest.main(['--doctest-modules', __file__, tested_module.__file__])
