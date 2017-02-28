# -*- coding: utf-8 -*-
import inspect
import itertools
from unittest.mock import Mock

import pytest
from multiset import Multiset

from matchpy.expressions.substitution import Substitution
from .common import *


class TestSubstitution:
    @pytest.mark.parametrize(
        '   substitution,                   variable,   value,                  expected_result',
        [
            ({},                            'x',        a,                      {'x': a}),
            ({'x': a},                      'x',        a,                      {'x': a}),
            ({'x': a},                      'x',        b,                      ValueError),
            ({'x': a},                      'x',        (a, b),                 ValueError),
            ({'x': (a, )},                  'x',        a,                      ValueError),
            ({'x': (a, b)},                 'x',        (a, b),                 {'x': (a, b)}),
            ({'x': (a, b)},                 'x',        (a, a),                 ValueError),
            ({'x': (a, b)},                 'x',        Multiset([a, b]),       {'x': (a, b)}),
            ({'x': (a, b)},                 'x',        Multiset([a]),          ValueError),
            ({'x': Multiset([a, b])},       'x',        Multiset([a, b]),       {'x': Multiset([a, b])}),
            ({'x': Multiset([a, b])},       'x',        Multiset([]),           ValueError),
            ({'x': Multiset([a, b])},       'x',        (a, b),                 {'x': (a, b)}),
            ({'x': Multiset([a, b])},       'x',        (a, a),                 ValueError),
            ({'x': Multiset([a])},          'x',        (a, ),                  {'x': (a, )}),
            ({'x': Multiset([a])},          'x',        (b, ),                  ValueError),
            ({'x': Multiset([a])},          'x',        a,                      ValueError),
            ({'x': Multiset([a])},          'x',        b,                      ValueError),
        ]
    )  # yapf: disable
    def test_union_with_var(self, substitution, variable, value, expected_result):
        substitution = Substitution(substitution)
        if expected_result is ValueError:
            with pytest.raises(ValueError):
                _ = substitution.union_with_variable(variable, value)
        else:
            result = substitution.union_with_variable(variable, value)
            assert result == expected_result

    @pytest.mark.parametrize(
        '   substitution1,                  substitution2,                  expected_result',
        [
            ({},                            {},                             {}),
            ({'x': a},                      {},                             {'x': a}),
            ({'x': a},                      {'y': b},                       {'x': a, 'y': b}),
            ({'x': a},                      {'x': b},                       ValueError),
            ({'x': a},                      {'x': a},                       {'x': a}),
        ]
    )  # yapf: disable
    def test_union(self, substitution1, substitution2, expected_result):
        substitution1 = Substitution(substitution1)
        substitution2 = Substitution(substitution2)
        if expected_result is ValueError:
            with pytest.raises(ValueError):
                _ = substitution1.union(substitution2)
            with pytest.raises(ValueError):
                _ = substitution2.union(substitution1)
        else:
            result = substitution1.union(substitution2)
            assert result == expected_result
            assert result is not substitution1
            assert result is not substitution2
            result = substitution2.union(substitution1)
            assert result == expected_result
            assert result is not substitution1
            assert result is not substitution2

    @pytest.mark.parametrize(
        '   substitution,                   subject,    pattern,                expected_result',
        [
            ({},                            a,          a,                      {}),
            ({},                            a,          x_,                     {'x': a}),
            ({'x': a},                      a,          x_,                     {'x': a}),
            ({'x': b},                      a,          x_,                     False),
            ({},                            f(a),       f(a),                   {}),
            ({},                            f(a),       f(x_),                  {'x': a}),
            ({'x': a},                      f(a),       f(x_),                  {'x': a}),
            ({'x': b},                      f(a),       f(x_),                  False),
            ({},                            f(a, a),    f(x_, x_),              {'x': a}),
            ({},                            f(a, b),    f(x_, x_),              False),
            ({},                            f(a, b),    f(x_, y_),              {'x': a, 'y': b}),
        ]
    )  # yapf: disable
    def test_extract_substitution(self, substitution, subject, pattern, expected_result):
        substitution = Substitution(substitution)
        if expected_result is False:
            assert substitution.extract_substitution(subject, pattern) is False
        else:
            assert substitution.extract_substitution(subject, pattern) is True
            assert substitution == expected_result

    @pytest.mark.parametrize(
        '   substitution,                  renaming,                  expected_result',
        [
            ({},                            {},                       {}),
            ({'x': a},                      {},                       {'x': a}),
            ({'x': a},                      {'x': 'y'},               {'y': a}),
            ({'x': a},                      {'y': 'x'},               {'x': a}),
        ]
    )  # yapf: disable
    def test_rename(self, substitution, renaming, expected_result):
        assert Substitution(substitution).rename(renaming) == expected_result

    def test_copy(self):
        substitution = Substitution({'x': a})

        copy = substitution.__copy__()

        assert copy == substitution
        assert copy is not substitution
