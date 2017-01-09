# -*- coding: utf-8 -*-
from unittest.mock import Mock

import pytest

from matchpy.expressions.constraints import (Constraint,
                                                     CustomConstraint,
                                                     EqualVariablesConstraint,
                                                     MultiConstraint)


class DummyConstraint(Constraint):
    def __call__(self, match):
        raise NotImplementedError()

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def with_renamed_vars(self, renaming):
        return self

C_dummy1 = DummyConstraint()
C_dummy2 = DummyConstraint()

MULTI_CONSTRAINTS = [
    MultiConstraint(),
    MultiConstraint(C_dummy2),
    MultiConstraint(C_dummy1),
    MultiConstraint(C_dummy2, C_dummy1),
]

VARIABLE_CONSTRAINTS = [
    EqualVariablesConstraint(),
    EqualVariablesConstraint('x'),
    EqualVariablesConstraint('y'),
    EqualVariablesConstraint('x', 'y'),
]

C_custom1 = CustomConstraint(lambda x: x == 0)
C_custom2 = CustomConstraint(lambda y: y == 0)
C_custom3 = CustomConstraint(lambda x, y: x == y)

CUSTOM_CONSTRAINTS = [C_custom1, C_custom2, C_custom3]


@pytest.mark.parametrize(
    '   constraints,              expected_result',
    [
        ([],                                                None),
        ([None],                                            None),
        ([C_dummy1],                                        C_dummy1),
        ([C_dummy1, C_dummy1],                              C_dummy1),
        ([C_dummy1, C_dummy2],                              MultiConstraint(C_dummy1, C_dummy2)),
        ([C_dummy1, None],                                  C_dummy1),
        ([C_dummy1, MultiConstraint(C_dummy1)],             C_dummy1),
        ([C_dummy1, MultiConstraint(C_dummy1, C_dummy2)],   MultiConstraint(C_dummy1, C_dummy2)),
    ]
)
def test_multi_constraint_create(constraints, expected_result):
    result = MultiConstraint.create(*constraints)

    assert result == expected_result


def test_multi_constraint_call():
    subst = object()

    c1 = Mock(return_value=False)
    c2 = Mock(return_value=True)

    mc = MultiConstraint(c1)
    assert mc(subst) is False
    c1.assert_called_once_with(subst)
    c1.reset_mock()

    mc = MultiConstraint(c2)
    assert mc(subst) is True
    c2.assert_called_once_with(subst)
    c2.reset_mock()

    mc = MultiConstraint(c1, c2)
    assert mc(subst) is False
    c1.assert_called_once_with(subst)


@pytest.mark.parametrize('c1', enumerate(MULTI_CONSTRAINTS))
@pytest.mark.parametrize('c2', enumerate(MULTI_CONSTRAINTS))
def test_multi_constraint_hash(c1, c2):
    i, c1 = c1
    j, c2 = c2
    if i == j:
        assert c1 == c2
        assert hash(c1) == hash(c2)
    else:
        assert c1 != c2
        assert hash(c1) != hash(c2)


@pytest.mark.parametrize(
    '   variables,      substitution,           expected_result',
    [
        (['x', 'y'],    {'x': 0, 'y': 0},       True),
        (['x', 'y'],    {'x': 0, 'y': 1},       False),
    ]
)
def test_equal_variables_constraint_call(variables, substitution, expected_result):
    constraint = EqualVariablesConstraint(*variables)
    result = constraint(substitution)
    assert result == expected_result


@pytest.mark.parametrize('c1', enumerate(VARIABLE_CONSTRAINTS))
@pytest.mark.parametrize('c2', enumerate(VARIABLE_CONSTRAINTS))
def test_equal_variables_constraint_hash(c1, c2):
    i, c1 = c1
    j, c2 = c2
    if i == j:
        assert c1 == c2
        assert hash(c1) == hash(c2)
    else:
        assert c1 != c2
        assert hash(c1) != hash(c2)


@pytest.mark.parametrize(
    '   constraint,     substitution,               expected_result',
    [
        (C_custom1,     {'x': 0, 'y': 0},           True),
        (C_custom1,     {'x': 1, 'y': 0},           False),
        (C_custom2,     {'x': 0, 'y': 0},           True),
        (C_custom2,     {'x': 0, 'y': 1},           False),
        (C_custom3,     {'x': 0, 'y': 0},           True),
        (C_custom3,     {'x': 0, 'y': 1},           False),
        (C_custom3,     {'x': 1, 'y': 0},           False),
        (C_custom3,     {'x': 1, 'y': 1},           True),
    ]
)
def test_custom_constraint_call(constraint, substitution, expected_result):
    result = constraint(substitution)
    assert result == expected_result


@pytest.mark.parametrize('c1', enumerate(CUSTOM_CONSTRAINTS))
@pytest.mark.parametrize('c2', enumerate(CUSTOM_CONSTRAINTS))
def test_custom_constraint_hash(c1, c2):
    i, c1 = c1
    j, c2 = c2
    if i == j:
        assert c1 == c2
        assert hash(c1) == hash(c2)
    else:
        assert c1 != c2
        assert hash(c1) != hash(c2)


def test_custom_constraint_errors():
    with pytest.raises(ValueError):
        CustomConstraint(lambda *args: True)
    with pytest.raises(ValueError):
        CustomConstraint(lambda **kwargs: True)


def test_equal_variables_constraint_with_renamed_vars():
    c1 = EqualVariablesConstraint('x', 'y')
    c2 = c1.with_renamed_vars({'x': 'z'})

    assert c2({'x': 1, 'z': 2, 'y': 1}) is False
    assert c2({'x': 1, 'z': 2, 'y': 2}) is True


def test_multi_constraint_with_renamed_vars():
    c1 = CustomConstraint(lambda x: True)
    c2 = CustomConstraint(lambda y: True)
    renaming = {'x': 'z', 'y': 'z2'}
    c1_renamed = c1.with_renamed_vars(renaming)
    c2_renamed = c2.with_renamed_vars(renaming)
    mc = MultiConstraint(c1, c2)
    mc_renamed = mc.with_renamed_vars(renaming)

    assert mc_renamed == MultiConstraint(c1_renamed, c2_renamed)


def test_custom_constraint_with_renamed_vars():
    actual_x = None
    actual_y = None
    def constraint(x, y):
        nonlocal actual_x
        nonlocal actual_y
        actual_x = x
        actual_y = y

        return x == y

    c1 = CustomConstraint(constraint)
    c2 = c1.with_renamed_vars({'x': 'z'})

    assert c2({'x': 1, 'z': 2, 'y': 1}) is False
    assert actual_x == 2
    assert actual_y == 1
    assert c2({'x': 1, 'z': 3, 'y': 3}) is True
    assert actual_x == 3
    assert actual_y == 3
