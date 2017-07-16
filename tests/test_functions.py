# -*- coding: utf-8 -*-
from hypothesis import assume, given
import hypothesis.strategies as st
import pytest

from matchpy.expressions.expressions import Arity, Operation, Symbol, Wildcard, Pattern
from matchpy.functions import ReplacementRule, replace, replace_all, substitute, replace_many, is_match
from matchpy.matching.one_to_one import match_anywhere
from matchpy.matching.one_to_one import match as match_one_to_one
from matchpy.matching.many_to_one import ManyToOneReplacer
from .common import *


@pytest.mark.parametrize(
    '   expr,       pattern,    do_match',
    [
        (a,         a,          True),
        (a,         b,          False),
        (f(a),      f(x_),      True),
    ]
)  # yapf: disable
def test_is_match(expr, pattern, do_match):
    assert is_match(expr, Pattern(pattern)) == do_match


class TestSubstitute:
    @pytest.mark.parametrize(
        '   expression,                         substitution,           expected_result,    replaced',
        [
            (a,                                 {},                     a,                  False),
            (a,                                 {'x': b},               a,                  False),
            (x_,                                {'x': b},               b,                  True),
            (x_,                                {'x': [a, b]},          [a, b],             True),
            (y_,                                {'x': b},               y_,                 False),
            (f(x_),                             {'x': b},               f(b),               True),
            (f(x_),                             {'y': b},               f(x_),              False),
            (f(x_),                             {},                     f(x_),              False),
            (f(a, x_),                          {'x': b},               f(a, b),            True),
            (f(x_),                             {'x': [a, b]},          f(a, b),            True),
            (f(x_),                             {'x': []},              f(),                True),
            (f(x_, c),                          {'x': [a, b]},          f(a, b, c),         True),
            (f(x_, y_),                         {'x': a, 'y': b},       f(a, b),            True),
            (f(x_, y_),                         {'x': [a, c], 'y': b},  f(a, c, b),         True),
            (f(x_, y_),                         {'x': a, 'y': [b, c]},  f(a, b, c),         True),
            (Pattern(f(x_)),                    {'x': a},               f(a),               True)
        ]
    )  # yapf: disable
    def test_substitute(self, expression, substitution, expected_result, replaced):
        result = substitute(expression, substitution)
        assert result == expected_result, "Substitution did not yield expected result"
        if replaced:
            assert result is not expression, "When substituting, the original expression may not be modified"
        else:
            assert result is expression, "When nothing is substituted, the original expression has to be returned"


def many_replace_wrapper(expression, position, replacement):
    return replace_many(expression, [(position, replacement)])


class TestReplaceTest:
    @pytest.mark.parametrize('replace', [replace, many_replace_wrapper])
    @pytest.mark.parametrize(
        '   expression,             position,   replacement,    expected_result',
        [
            (a,                     (),         b,              b),
            (f(a),                  (),         b,              b),
            (a,                     (),         f(b),           f(b)),
            (f(a),                  (),         f(b),           f(b)),
            (f(a),                  (0, ),      b,              f(b)),
            (f(a, b),               (0, ),      c,              f(c, b)),
            (f(a, b),               (1, ),      c,              f(a, c)),
            (f(a),                  (0, ),      [b, c],         f(b, c)),
            (f(a, b),               (0, ),      [b, c],         f(b, c, b)),
            (f(a, b),               (1, ),      [b, c],         f(a, b, c)),
            (f(f(a)),               (0, ),      b,              f(b)),
            (f(f(a)),               (0, 0),     b,              f(f(b))),
            (f(f(a, b)),            (0, 0),     c,              f(f(c, b))),
            (f(f(a, b)),            (0, 1),     c,              f(f(a, c))),
            (f(f(a, b), f(a, b)),   (0, 0),     c,              f(f(c, b), f(a, b))),
            (f(f(a, b), f(a, b)),   (0, 1),     c,              f(f(a, c), f(a, b))),
            (f(f(a, b), f(a, b)),   (1, 0),     c,              f(f(a, b), f(c, b))),
            (f(f(a, b), f(a, b)),   (1, 1),     c,              f(f(a, b), f(a, c))),
            (f(f(a, b), f(a, b)),   (0, ),      c,              f(c, f(a, b))),
            (f(f(a, b), f(a, b)),   (1, ),      c,              f(f(a, b), c)),
        ]
    )  # yapf: disable
    def test_substitution_match(self, replace, expression, position, replacement, expected_result):
        result = replace(expression, position, replacement)
        assert result == expected_result, "Replacement did not yield expected result ({!r} {!r} -> {!r})".format(
            expression, position, replacement
        )
        assert result is not expression, "Replacement modified the original expression"

    @pytest.mark.parametrize('replace', [replace, many_replace_wrapper])
    def test_too_big_position_error(self, replace):
        with pytest.raises(IndexError):
            replace(a, (0, ), b)
        with pytest.raises(IndexError):
            replace(f(a), (0, 0), b)
        with pytest.raises(IndexError):
            replace(f(a), (1, ), b)
        with pytest.raises(IndexError):
            replace(f(a, b), (2, ), b)


class TestReplaceManyTest:
    @pytest.mark.parametrize(
        '   expression,             replacements,                           expected_result',
        [
            (f(a, b),               [((0, ),  b), ((1, ),  a)],             f(b, a)),
            (f(a, b),               [((0, ),  [c, c]), ((1, ),  a)],        f(c, c, a)),
            (f(a, b),               [((0, ),  b), ((1, ),  [c, c])],        f(b, c, c)),
            (f(f2(a, b), c),        [((0, 0),  b), ((0, 1),  a)],           f(f2(b, a), c)),
            (f_c(c, f2(a, b)),       [((1, 0),  b), ((1, 1),  a)],           f_c(c, f2(b, a))),
            (f(f2(a, b), f2(c)),    [((1, 0),  b), ((0, 1),  a)],           f(f2(a, a), f2(b))),
            (f(f2(a, b), f2(c)),    [((0, 1),  a), ((1, 0),  b)],           f(f2(a, a), f2(b))),
            (f_c(f2(c), f2(a, b)),   [((0, 0),  b), ((1, 1),  a)],           f_c(f2(b), f2(a, a))),
            (f_c(f2(c), f2(a, b)),   [((1, 1),  a), ((0, 0),  b)],           f_c(f2(b), f2(a, a))),
        ]
    )  # yapf: disable
    def test_substitution_match(self, expression, replacements, expected_result):
        result = replace_many(expression, replacements)
        assert result == expected_result, "Replacement did not yield expected result ({!r} -> {!r})".format(
            expression, replacements
        )
        assert result is not expression, "Replacement modified the original expression"

    def test_inconsistent_position_error(self):
        with pytest.raises(IndexError):
            replace_many(f(a), [((), b), ((0, ), b)])
        with pytest.raises(IndexError):
            replace_many(a, [((), b), ((0, ), b)])
        with pytest.raises(IndexError):
            replace_many(a, [((0, ), b), ((1, ), b)])

    def test_empty_replace(self):
        expression = f(a, b)
        result = replace_many(expression, [])
        assert expression is result, "Empty replacements should not change the expression."


@pytest.mark.parametrize(
    '   expression,                                             pattern,    expected_results',
    [                                                                       # Substitution      Position
        (f(a),                                                  f(x_),      [({'x': a},         ())]),
        (f(a),                                                  x_,         [({'x': f(a)},      ()),
                                                                             ({'x': a},         (0, ))]),
        (f(a, f2(b), f2(f2(c), f2(a), f2(f2(b))), f2(c), c),    f2(x_),     [({'x': b},         (1, )),
                                                                             ({'x': c},         (2, 0)),
                                                                             ({'x': a},         (2, 1)),
                                                                             ({'x': f2(b)},     (2, 2)),
                                                                             ({'x': b},         (2, 2, 0)),
                                                                             ({'x': c},         (3, ))])
    ]
)  # yapf: disable
def test_match_anywhere(expression, pattern, expected_results):
    expression = expression
    pattern = Pattern(pattern)
    results = list(match_anywhere(expression, pattern))

    assert len(results) == len(expected_results), "Invalid number of results"

    for result in expected_results:
        assert result in results, "Results differ from expected"


def test_match_anywhere_error():
    with pytest.raises(ValueError):
        next(match_anywhere(f(x_), f(x_)))


def test_match_error():
    with pytest.raises(ValueError):
        next(match_one_to_one(f(x_), f(x_)))


def _many_to_one_replace(expression, rules):
    return ManyToOneReplacer(*rules).replace(expression)

@pytest.mark.parametrize(
    'replacer', [replace_all, _many_to_one_replace]
)
def test_logic_simplify(replacer):
    LAnd = Operation.new('and', Arity.variadic, 'LAnd', associative=True, one_identity=True, commutative=True)
    LOr = Operation.new('or', Arity.variadic, 'LOr', associative=True, one_identity=True, commutative=True)
    LXor = Operation.new('xor', Arity.variadic, 'LXor', associative=True, one_identity=True, commutative=True)
    LNot = Operation.new('not', Arity.unary, 'LNot')
    LImplies = Operation.new('implies', Arity.binary, 'LImplies')
    Iff = Operation.new('iff', Arity.binary, 'Iff')

    ___ = Wildcard.star()

    a1 = Symbol('a1')
    a2 = Symbol('a2')
    a3 = Symbol('a3')
    a4 = Symbol('a4')
    a5 = Symbol('a5')
    a6 = Symbol('a6')
    a7 = Symbol('a7')
    a8 = Symbol('a8')
    a9 = Symbol('a9')
    a10 = Symbol('a10')
    a11 = Symbol('a11')

    LBot = Symbol(u'⊥')
    LTop = Symbol(u'⊤')

    expression = LImplies(
        LAnd(
            Iff(
                Iff(LOr(a1, a2), LOr(LNot(a3), Iff(LXor(a4, a5), LNot(LNot(LNot(a6)))))),
                LNot(
                    LAnd(
                        LAnd(a7, a8),
                        LNot(
                            LXor(
                                LXor(LOr(a9, LAnd(a10, a11)), a2),
                                LAnd(LAnd(a11, LXor(a2, Iff(a5, a5))), LXor(LXor(a7, a7), Iff(a9, a4)))
                            )
                        )
                    )
                )
            ),
            LImplies(
                Iff(
                    Iff(LOr(a1, a2), LOr(LNot(a3), Iff(LXor(a4, a5), LNot(LNot(LNot(a6)))))),
                    LNot(
                        LAnd(
                            LAnd(a7, a8),
                            LNot(
                                LXor(
                                    LXor(LOr(a9, LAnd(a10, a11)), a2),
                                    LAnd(LAnd(a11, LXor(a2, Iff(a5, a5))), LXor(LXor(a7, a7), Iff(a9, a4)))
                                )
                            )
                        )
                    )
                ),
                LNot(
                    LAnd(
                        LImplies(
                            LAnd(a1, a2),
                            LNot(
                                LXor(
                                    LOr(
                                        LOr(
                                            LXor(LImplies(LAnd(a3, a4), LImplies(a5, a6)), LOr(a7, a8)),
                                            LXor(Iff(a9, a10), a11)
                                        ), LXor(LXor(a2, a2), a7)
                                    ), Iff(LOr(a4, a9), LXor(LNot(a6), a6))
                                )
                            )
                        ), LNot(Iff(LNot(a11), LNot(a9)))
                    )
                )
            )
        ),
        LNot(
            LAnd(
                LImplies(
                    LAnd(a1, a2),
                    LNot(
                        LXor(
                            LOr(
                                LOr(
                                    LXor(LImplies(LAnd(a3, a4), LImplies(a5, a6)), LOr(a7, a8)),
                                    LXor(Iff(a9, a10), a11)
                                ), LXor(LXor(a2, a2), a7)
                            ), Iff(LOr(a4, a9), LXor(LNot(a6), a6))
                        )
                    )
                ), LNot(Iff(LNot(a11), LNot(a9)))
            )
        )
    )

    rules = [
        # xor(x,⊥) → x
        ReplacementRule(
            Pattern(LXor(x__, LBot)),
            lambda x: LXor(*x)
        ),
        # xor(x, x) → ⊥
        ReplacementRule(
            Pattern(LXor(x_, x_, ___)),
            lambda x: LBot
        ),
        # and(x,⊤) → x
        ReplacementRule(
            Pattern(LAnd(x__, LTop)),
            lambda x: LAnd(*x)
        ),
        # and(x,⊥) → ⊥
        ReplacementRule(
            Pattern(LAnd(__, LBot)),
            lambda: LBot
        ),
        # and(x, x) → x
        ReplacementRule(
            Pattern(LAnd(x_, x_, y___)),
            lambda x, y: LAnd(x, *y)
        ),
        # and(x, xor(y, z)) → xor(and(x, y), and(x, z))
        ReplacementRule(
            Pattern(LAnd(x_, LXor(y_, z_))),
            lambda x, y, z: LXor(LAnd(x, y), LAnd(x, z))
        ),
        # implies(x, y) → not(xor(x, and(x, y)))
        ReplacementRule(
            Pattern(LImplies(x_, y_)),
            lambda x, y: LNot(LXor(x, LAnd(x, y)))
        ),
        # not(x) → xor(x,⊤)
        ReplacementRule(
            Pattern(LNot(x_)),
            lambda x: LXor(x, LTop)
        ),
        # or(x, y) → xor(and(x, y), xor(x, y))
        ReplacementRule(
            Pattern(LOr(x_, y_)),
            lambda x, y: LXor(LAnd(x, y), LXor(x, y))
        ),
        # iff(x, y) → not(xor(x, y))
        ReplacementRule(
            Pattern(Iff(x_, y_)),
            lambda x, y: LNot(LXor(x, y))
        ),
    ]  # yapf: disable

    result = replacer(expression, rules)

    assert result == LBot
