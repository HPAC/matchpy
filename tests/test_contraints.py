# -*- coding: utf-8 -*-
from unittest.mock import Mock

import pytest

from matchpy.expressions.constraints import Constraint, CustomConstraint, EqualVariablesConstraint


class DummyConstraint(Constraint):
    def __call__(self, match):
        raise NotImplementedError()

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


C_dummy1 = DummyConstraint()
C_dummy2 = DummyConstraint()

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
    '   variables,      substitution,           expected_result',
    [
        (['x', 'y'],    {'x': 0, 'y': 0},       True),
        (['x', 'y'],    {'x': 0, 'y': 1},       False),
    ]
)  # yapf: disable
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
)  # yapf: disable
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


def test_constraint_vars():
    assert len(Constraint().variables) == 0


def test_equal_variables_constraint_vars():
    c1 = EqualVariablesConstraint('x', 'y')

    assert c1.variables == {'x', 'y'}


def test_equal_variables_constraint_with_renamed_vars():
    c1 = EqualVariablesConstraint('x', 'y')
    c2 = c1.with_renamed_vars({'x': 'z'})

    assert c2({'x': 1, 'z': 2, 'y': 1}) is False
    assert c2({'x': 1, 'z': 2, 'y': 2}) is True


def test_custom_constraint_vars():
    c1 = CustomConstraint(lambda x, y: True)
    assert c1.variables == {'x', 'y'}


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
