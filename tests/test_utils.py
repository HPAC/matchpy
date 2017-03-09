# -*- coding: utf-8 -*-
import itertools
import os

from hypothesis import assume, example, given
import hypothesis.strategies as st
import pytest
from multiset import Multiset

from matchpy.utils import (
    VariableWithCount, base_solution_linear, cached_property, commutative_sequence_variable_partition_iter,
    extended_euclid, fixed_integer_vector_iter, get_short_lambda_source, weak_composition_iter, slot_cached_property,
    solve_linear_diop
)


def is_unique_list(l):
    for i, v1 in enumerate(l):
        for v2 in l[:i]:
            if v1 == v2:
                return False
    return True


class TestFixedIntegerVectorIterator:
    @given(st.lists(st.integers(min_value=0, max_value=10), max_size=5), st.integers(50))
    def test_correctness(self, vector, length):
        vector = tuple(vector)

        for result in fixed_integer_vector_iter(vector, length):
            assert sum(result) == length, '{}, {}, {}'.format(vector, length, result)


@given(st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=1000))
def test_extended_euclid(a, b):
    x, y, d = extended_euclid(a, b)
    assert a % d == 0
    assert b % d == 0
    assert a * x + b * y == d


class TestBaseSolutionLinear:
    @given(
        st.integers(min_value=1, max_value=1000),
        st.integers(min_value=1, max_value=1000),
        st.integers(min_value=0, max_value=1000),
    )
    def test_correctness(self, a, b, c):
        for x, y in base_solution_linear(a, b, c):
            assert x >= 0
            assert y >= 0
            assert a * x + b * y == c, "Invalid solution {!r},{!r}".format(x, y)

    @given(
        st.integers(min_value=1, max_value=1000),
        st.integers(min_value=1, max_value=1000),
        st.integers(min_value=0, max_value=1000),
    )
    def test_completeness(self, a, b, c):
        solutions = set(base_solution_linear(a, b, c))
        for x in range(c + 1):
            for y in range(c - a * x):
                if a * x + b * y == c:
                    assert (x, y) in solutions, "Missing solution {!r},{!r}".format(x, y)

    @given(
        st.integers(min_value=1, max_value=1000),
        st.integers(min_value=1, max_value=1000),
        st.integers(min_value=0, max_value=1000),
    )
    def test_uniqueness(self, a, b, c):
        solutions = list(base_solution_linear(a, b, c))
        assert is_unique_list(solutions), "Duplicate solution found"

    @pytest.mark.parametrize(
        '   a,      b,      c',
        [
            (0,     1,      1),
            (-1,    1,      1),
            (1,     0,      1),
            (1,     -1,     1),
            (1,     1,      -1),
        ]
    )  # yapf: disable
    def test_error(self, a, b, c):
        with pytest.raises(ValueError):
            next(base_solution_linear(a, b, c))


class TestIntegerPartitionVectorIter:
    @pytest.mark.parametrize('n', range(0, 11))
    @pytest.mark.parametrize('m', range(0, 4))
    def test_correctness(self, n, m):
        for part in weak_composition_iter(n, m):
            assert all(p >= 0 for p in part)
            assert sum(part) == n
            assert len(part) == m

    @pytest.mark.parametrize('n', range(0, 11))
    @pytest.mark.parametrize('m', range(0, 4))
    def test_completeness_and_uniqueness(self, n, m):
        solutions = set(weak_composition_iter(n, m))

        if m == 0 and n > 0:
            expected_count = 0
        else:
            # the total number of distinct partitions is given by (n+m-1)!/((m-1)!*n!)
            expected_count = 1
            for i in range(1, m):
                expected_count *= n + m - i
            for i in range(1, m):
                expected_count /= i

        assert len(solutions) == expected_count
        assert len(set(solutions)) == expected_count

    def test_error(self):
        with pytest.raises(ValueError):
            next(weak_composition_iter(-1, 1))
        with pytest.raises(ValueError):
            next(weak_composition_iter(1, -1))


class TestSolveLinearDiop:
    @staticmethod
    def _limit_possible_solution_count(coeffs, c):
        total_solutions_approx = 1
        for coeff in coeffs:
            if c % coeff == 0:
                total_solutions_approx *= c / coeff
        assume(total_solutions_approx <= 100)

    @given(st.lists(st.integers(min_value=1, max_value=100), max_size=5), st.integers(min_value=0, max_value=100))
    @example([1, 2, 2], 4)
    def test_correctness(self, coeffs, c):
        self._limit_possible_solution_count(coeffs, c)
        for solution in solve_linear_diop(c, *coeffs):
            assert len(solution) == len(coeffs), "Solution size differs from coefficient count"
            result = sum(c * x for c, x in zip(coeffs, solution))
            for x in solution:
                assert x >= 0
            assert result == c, "Invalid solution {!r}".format(solution)

    @given(st.lists(st.integers(min_value=1, max_value=100), max_size=5), st.integers(min_value=0, max_value=100))
    @example([1, 2, 2], 4)
    def test_completeness(self, coeffs, c):
        self._limit_possible_solution_count(coeffs, c)
        solutions = set(solve_linear_diop(c, *coeffs))
        values = [range(c // x) for x in coeffs]
        for solution2 in itertools.product(*values):
            result = sum(c * x for c, x in zip(coeffs, solution2))
            if result == c:
                assert solution2 in solutions, "Missing solution {!r}".format(solution2)

    @given(st.lists(st.integers(min_value=1, max_value=100), max_size=5), st.integers(min_value=0, max_value=100))
    @example([1, 2, 2], 4)
    def test_uniqueness(self, coeffs, c):
        self._limit_possible_solution_count(coeffs, c)
        solutions = list(solve_linear_diop(c, *coeffs))
        assert is_unique_list(solutions), "Duplicate solution found"


@st.composite
def sequence_vars(draw):
    num_vars = draw(st.integers(min_value=1, max_value=4))

    variables = []
    for i in range(num_vars):
        name = 'var{:d}'.format(i)
        count = draw(st.integers(min_value=1, max_value=4))
        minimum = draw(st.integers(min_value=0, max_value=2))

        variables.append(VariableWithCount(name, count, minimum))

    return variables


class TestCommutativeSequenceVariablePartitionIter:
    @given(sequence_vars(), st.lists(st.integers(1, 4), min_size=1, max_size=10))
    def test_correctness_randomized(self, variables, values):
        values = Multiset(values)
        for subst in commutative_sequence_variable_partition_iter(values, variables):
            assert len(variables) == len(subst)
            result_union = Multiset()
            for var in variables:
                assert len(subst[var.name]) >= var.minimum
                result_union.update(subst[var.name] * var.count)
            assert result_union == values

    @pytest.mark.parametrize(
        '   variables,              values,     expected_iter_count',
        #   Variables have the form (count, minimum length)
        [
            ([],                    'a',        0),
            ([],                    '',         1),
            ([(1, 0)],              '',         1),
            ([(1, 0)],              'a',        1),
            ([(1, 1)],              '',         0),
            ([(1, 1)],              'a',        1),
            ([(1, 2)],              'a',        0),
            ([(2, 0)],              '',         1),
            ([(2, 0)],              'a',        0),
            ([(2, 1)],              '',         0),
            ([(2, 1)],              'a',        0),
            ([(2, 2)],              'a',        0),
            ([(2, 0)],              'ab',       0),
            ([(2, 1)],              'ab',       0),
            ([(2, 2)],              'ab',       0),
            ([(2, 0)],              'aa',       1),
            ([(2, 1)],              'aa',       1),
            ([(2, 2)],              'aa',       0),
            ([(2, 0)],              'aaa',      0),
            ([(2, 1)],              'aaa',      0),
            ([(2, 2)],              'aaa',      0),
            ([(2, 0)],              'aabb',     1),
            ([(2, 1)],              'aabb',     1),
            ([(2, 2)],              'aabb',     1),
            ([(1, 0), (1, 0)],      '',         1),
            ([(1, 0), (1, 0)],      'a',        2),
            ([(1, 1), (1, 0)],      '',         0),
            ([(1, 1), (1, 0)],      'a',        1),
            ([(1, 0), (1, 0)],      'aa',       3),
            ([(1, 1), (1, 0)],      'aa',       2),
            ([(1, 0), (1, 0)],      'ab',       4),
            ([(1, 1), (1, 0)],      'ab',       3),
            ([(1, 0), (1, 0)],      'aaa',      4),
            ([(1, 1), (1, 0)],      'aaa',      3),
            ([(1, 0), (1, 0)],      'aab',      6),
            ([(1, 1), (1, 0)],      'aab',      5),
            ([(1, 0), (1, 0)],      'a',        2),
            ([(1, 1), (1, 0)],      'a',        1),
            ([(2, 0), (1, 0)],      '',         1),
            ([(2, 0), (1, 0)],      'aa',       2),
            ([(2, 1), (1, 0)],      '',         0),
            ([(2, 1), (1, 0)],      'aa',       1),
            ([(2, 0), (1, 0)],      'ab',       1),
            ([(2, 1), (1, 0)],      'ab',       0),
            ([(2, 0), (1, 0)],      'aaa',      2),
            ([(2, 1), (1, 0)],      'aaa',      1),
            ([(2, 0), (1, 0)],      'aab',      2),
            ([(2, 1), (1, 0)],      'aab',      1),
        ]
    )  # yapf: disable
    def test_correctness(self, variables, values, expected_iter_count):
        values = Multiset(values)
        variables = [VariableWithCount('var{:d}'.format(i), c, m) for i, (c, m) in enumerate(variables)]
        count = 0
        for subst in commutative_sequence_variable_partition_iter(values, variables):
            assert len(variables) == len(subst), "Wrong number of variables in the substitution"
            result_union = Multiset()
            for var in variables:
                assert len(subst[var.name]) >= var.minimum, "Variable did not get its minimum number of expressions"
                result_union.update(subst[var.name] * var.count)
            assert result_union == values, "Substitution is not a partition of the values"
            count += 1
        assert count == expected_iter_count, "Invalid number of substitution in the iterable"


# yapf: disable
# =========================================================================
# DON'T CHANGE THE FORMATTING OF THESE LINES, IT IS IMPORTANT FOR THE TESTS
lambda_example = lambda a: a > 0, 0
def not_a_lambda(): pass
lambda_multiline_example = [
    lambda x, y: x == y
]
# =========================================================================
# yapf: enable

@pytest.mark.parametrize(
    '   lambda_func,                    expected_source',
    [
        ((lambda x: x == 1),            'x == 1'),
        (lambda x: x == 1,              'x == 1'),
        (lambda_example[0],             'a > 0'),
        (lambda_multiline_example[0],   'x == y'),
        (lambda x: x[0],                'x[0]'),
        (lambda x: x == (0, 1),         'x == (0, 1)'),
        # FORMATTING IS IMPORTANT HERE TOO
        (lambda x: x > 1 and \
                   x < 5,               'x > 1 and \\{}x < 5'.format(os.linesep)),
        ([lambda x: x == 42, 5][0],     'x == 42'),
        (not_a_lambda,                  None),
        (5,                             None),
    ]
)  # yapf: disable
def test_get_short_lambda_source(lambda_func, expected_source):
    source = get_short_lambda_source(lambda_func)
    assert source == expected_source


def test_cached_property():
    class A:
        call_count = 0

        @cached_property
        def example(self):
            """Docstring Test"""
            A.call_count += 1
            return 42

    a = A()
    b = A()

    assert A.call_count == 0
    assert a.example == 42
    assert A.call_count == 1
    assert a.example == 42
    assert A.call_count == 1
    assert b.example == 42
    assert A.call_count == 2
    assert b.example == 42
    assert A.call_count == 2
    assert A.example.__doc__ == "Docstring Test"


def test_slot_cached_property():
    class A:
        __slots__ = ('cache', )
        call_count = 0

        @slot_cached_property('cache')
        def example(self):
            """Docstring Test"""
            A.call_count += 1
            return 42

    a = A()
    b = A()

    assert A.call_count == 0
    assert a.example == 42
    assert A.call_count == 1
    assert a.example == 42
    assert A.call_count == 1
    assert b.example == 42
    assert A.call_count == 2
    assert b.example == 42
    assert A.call_count == 2
    assert A.example.__doc__ == "Docstring Test"
