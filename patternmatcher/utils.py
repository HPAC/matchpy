# -*- coding: utf-8 -*-
import inspect
import math
import re
import ast
import os
from typing import (Callable, Dict, Iterator,  # pylint: disable=unused-import
                    List, NamedTuple, Optional, Sequence, Tuple, TypeVar, cast)

from multiset import Multiset

T = TypeVar('T')


VariableWithCount = NamedTuple('VariableWithCount', [('name', str), ('count', int), ('minimum', int)])


def fixed_integer_vector_iter(maxVect: Tuple[int, ...], vector_sum: int) -> Iterator[Tuple[int, ...]]:
    """
    Return an iterator over the integer vectors which

    - are componentwise less than or equal to `maxVect`, and
    - are non-negative, and where
    - the sum of their components is exactly `vector_sum`.

    The iterator yields the vectors in lexicographical order.

    Examples:

        List all vectors that are between (0, 0) and (2, 2) componentwise, where the sum of components is 2:

        >>> vectors = list(fixed_integer_vector_iter([2, 2], 2))
        >>> vectors
        [(0, 2), (1, 1), (2, 0)]
        >>> list(map(sum, vectors))
        [2, 2, 2]
    """
    if len(maxVect) == 0:
        yield tuple()
        return

    total = sum(maxVect)
    if vector_sum <= total:
        vector_sum = min(vector_sum, total)
        start = max(maxVect[0] + vector_sum - total, 0)
        end = min(maxVect[0], vector_sum)

        for j in range(start, end + 1):
            for vec in fixed_integer_vector_iter(maxVect[1:], vector_sum - j):
                yield (j, ) + vec


def minimum_integer_vector_iter(maxVect: Tuple[int, ...], minSum: int=0) -> Iterator[Tuple[int, ...]]:
    if len(maxVect) == 0:
        yield tuple()
        return

    total = sum(maxVect)
    if minSum <= total:
        start = max(maxVect[0] + minSum - total, 0)

        for j in range(start, maxVect[0] + 1):
            newmin = max(0, minSum - j)
            for vec in minimum_integer_vector_iter(maxVect[1:], newmin):
                yield (j, ) + vec


def integer_partition_vector_iter(n: int, m: int) -> Iterator[List[int]]:
    """

    """
    if m < 0:
        return
    if m == 0:
        if n == 0:
            yield tuple()
        return
    if m == 1:
        yield (n, )
        return

    for i in range(0, n + 1):
        for vec in integer_partition_vector_iter(n - i, m - 1):
            yield (i, ) + vec


def fixed_sum_vector_iter(min_vect: Sequence[int], max_vect: Sequence[int], total: int) -> Iterator[List[int]]:
    assert len(min_vect) == len(max_vect), "len(min_vect) != len(max_vect)"
    assert all(min_value <= max_value for min_value, max_value in zip(min_vect, max_vect)), "min_vect > max_vect"

    min_sum = sum(min_vect)
    max_sum = sum(max_vect)

    if min_sum > total or max_sum < total:
        return

    count = len(max_vect)

    if count <= 1:
        if len(max_vect) == 1:
            yield [total]
        else:
            yield []
        return

    remaining = total - min_sum

    real_mins = list(min_vect)
    real_maxs = list(max_vect)

    for i, (minimum, maximum) in enumerate(zip(min_vect, max_vect)):
        left_over_sum = sum(max_vect[:i]) + sum(max_vect[i+1:])
        if left_over_sum != math.inf:
            real_mins[i] = max(total - left_over_sum, minimum)
        real_maxs[i] = min(remaining + minimum, maximum)

    values = list(real_mins)

    remaining = total - sum(real_mins)

    if remaining == 0:
        yield values
        return

    j = count - 1
    while remaining > 0:
        to_add = min(real_maxs[j] - real_mins[j], remaining)
        values[j] += to_add
        remaining -= to_add
        j -= 1

    while True:
        pos = count - 2
        yield values[:]
        while True:
            values[pos] += 1
            values[-1] -= 1
            if values[-1] < real_mins[-1] or values[pos] > real_maxs[pos]:
                if pos == 0:
                    return
                variable_amount = values[pos] - real_mins[pos]
                values[pos] = real_mins[pos]  # reset current position
                values[-1] += variable_amount  # reset last position

                if values[-1] > real_maxs[-1]:
                    remaining = values[-1] - real_maxs[-1] - 1
                    values[-1] = real_maxs[-1] + 1
                    j = count - 2
                    while remaining > 0:
                        to_add = min(real_maxs[j] - values[j], remaining)
                        values[j] += to_add
                        remaining -= to_add
                        j -= 1
                pos -= 1
            else:
                break


def _make_iter_factory(value, total, variables: List[VariableWithCount]):
    var_counts = [v.count for v in variables]

    def factory(subst):
        for solution in solve_linear_diop(total, *var_counts):
            for var, count in zip(variables, solution):
                subst[var.name][value] = count
            yield (subst, )

    return factory


def commutative_sequence_variable_partition_iter(values: Multiset[T], variables: List[VariableWithCount]) \
        -> Iterator[Dict[str, Multiset[T]]]:
    iterators = []
    for value, count in values.items():
        iterators.append(_make_iter_factory(value, count, variables))

    initial = dict((var.name, Multiset()) for var in variables)  # type: Dict[str, Multiset[T]]

    for (subst, ) in iterator_chain((initial, ), *iterators):
        valid = True
        for var in variables:
            if len(subst[var.name]) < var.minimum:
                valid = False
                break
        if valid:
            if None in subst:
                del subst[None]
            yield subst


def get_short_lambda_source(lambda_func):
    """Return the source of a (short) lambda function.
    If it's impossible to obtain, returns None.
    """
    # Adapted from http://xion.io/post/code/python-get-lambda-code.html
    try:
        source_lines, _ = inspect.getsourcelines(lambda_func)
    except (IOError, TypeError):
        return None

    # Remove trailing whitespace from lines (including potential lingering \r and \n due to OS mismatch)
    source_lines = [l.rstrip() for l in source_lines]

    # Try to parse the source lines
    # In case we have an indentation error, wrap it in a compound statement
    try:
        source_ast = ast.parse(os.linesep.join(source_lines))
    except IndentationError:
        source_lines.insert(0, 'with 0:')
        source_ast = ast.parse(os.linesep.join(source_lines))

    # Find the first AST node that is a lambda definition
    lambda_node = next((node for node in ast.walk(source_ast)
                        if isinstance(node, ast.Lambda)), None)
    if lambda_node is None:  # It is a def fn(): ...
        return None

    # Remove everything before the first lambda's body
    # Remove indentation from lines
    lines = source_lines[lambda_node.lineno-1:]
    lines[0] = lines[0][lambda_node.body.col_offset:]
    lambda_body_text = os.linesep.join(l.lstrip() for l in lines)

    # Start with the full body and everything to the end of the source.
    # Start shaving away characters at the end until the source parses
    while True:
        try:
            code = compile(lambda_body_text, '<unused filename>', 'eval')

            # Check the size of the generated bytecode to avoid stopping the shaving too early:
            #
            #   bloop = lambda x: True, 0
            #
            # Here, "True, 0" is already a valid expression, but it is not the original lambda's body.
            # So compiling succeeds, but the bytecode doesn't check out
            # Also, the code is not compared directly, as the result might differ depending on the (global) context
            if len(code.co_code) == len(lambda_func.__code__.co_code):
                return lambda_body_text.strip()
        except SyntaxError:
            pass
        lambda_body_text = lambda_body_text[:-1]
        if not lambda_body_text:
            raise AssertionError # We (should) always get the valid body at some point


def extended_euclid(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm that computes the Bézout coefficients as well as `gcd(a, b)`

    Returns `x, y, d` where `x` and `y` are a solution to `ax + by = d` and `d = gcd(a, b)`.
    `x` and `y` are a minimal pair of Bézout's coefficients.

    See `Extended Euclidean algorithm <https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm>`_ or
    `Bézout's identity <https://en.wikipedia.org/wiki/B%C3%A9zout%27s_identity>`_ for more information.
    """
    if b == 0:
        return (1, 0, a)

    x0, y0, d = extended_euclid(b, a % b)
    x, y = y0, x0 - (a // b) * y0

    return (x, y, d)


def base_solution_linear(a: int, b: int, c: int) -> Iterator[Tuple[int, int]]:
    r"""Yields solution for a basic linear Diophantine equation of the form :math:`ax + by = c`.

    First, the equation is normalized by dividing :math:`a, b, c` by their gcd.
    Then, the extended Euclidean algorithm (:func:`extended_euclid`) is used to find a base solution :math:`(x_0, y_0)`.
    All non-negative solutions are generated by using that the general solution is:math:`(x_0 + b t, y_0 - a t)`.
    Hence, by adding or substracting :math:`a` resp. :math:`b` from the base solution, all solutions can be generated.
    Because the base solution is one of the minimal pairs of Bézout's coefficients, for all non-negative solutions
    either :math:`t \geq 0` or :math:`t \leq 0` must hold. Also, all the non-negative solutions are consecutive with
    respect to :math:`t`. Therefore, all non-negative solutions can be generated efficiently from the base solution.
    """
    assert a > 0, "Invalid coefficient"
    assert b > 0, "Invalid coefficient"

    d = math.gcd(a, math.gcd(b, c))
    a = a // d
    b = b // d
    c = c // d

    if c == 0:
        yield (0, 0)
    else:
        x0, y0, d = extended_euclid(a, b)

        # If c is not divisible by gcd(a, b), then there is no solution
        if c % d != 0:
            return

        x, y = c * x0, c * y0

        if x <= 0:
            while y >= 0:
                if x >= 0:
                    yield (x, y)
                x += b
                y -= a
        else:
            while x >= 0:
                if y >= 0:
                    yield (x, y)
                x -= b
                y += a


def solve_linear_diop(total: int, *coeffs: int) -> Iterator[Tuple[int, ...]]:
    r"""Generator for the solutions of a linear Diophantine equation of the form :math:`c_1 x_1 + \dots + c_n x_n = total`

    `coeffs` are the coefficients `c_i`.

    If there are at most two coefficients, :func:`base_solution_linear` is used to find the solutions.
    Otherwise, the solutions are found recursively, by reducing the number of variables in each recursion:

    1. Compute :math:`d := gcd(c_2, \dots , c_n)`
    2. Solve :math:`c_1 x + d y = total`
    3. Recursively solve :math:`c_2 x_2 + \dots + c_n x_n = y` for each solution for `y`
    4. Combine these solutions to form a solution for the whole equation
    """
    if len(coeffs) == 0:
        if total == 0:
            yield tuple()
        return
    if len(coeffs) == 1:
        if total % coeffs[0] == 0:
            yield (total // coeffs[0], )
        return
    if len(coeffs) == 2:
        yield from base_solution_linear(coeffs[0], coeffs[1], total)
        return

    # calculate gcd(coeffs[1:])
    remainder_gcd = math.gcd(coeffs[1], coeffs[2])
    for coeff in coeffs[3:]:
        remainder_gcd = math.gcd(remainder_gcd, coeff)

    # solve coeffs[0] * x + remainder_gcd * y = total
    for coeff0_solution, remainder_gcd_solution in base_solution_linear(coeffs[0], remainder_gcd, total):
        new_coeffs = [c // remainder_gcd for c in coeffs[1:]]
        # use the solutions for y to solve the remaining variables recursively
        for remainder_solution in solve_linear_diop(remainder_gcd_solution, *new_coeffs):
            yield (coeff0_solution, ) + remainder_solution


def _match_value_repr_str(value):  # pragma: no cover
    if isinstance(value, list):
        return '({!s})'.format(', '.join(str(x) for x in value))
    return str(value)


def match_repr_str(match):  # pragma: no cover
    return ', '.join('{!s}: {!s}'.format(k, _match_value_repr_str(v)) for k, v in match.items())


def is_sorted(l):
    for i, el in enumerate(l[1:]):
        if el > l[i]:
            return False
    return True


def iterator_chain(initial_data: tuple, *factories: Callable[..., Iterator[tuple]]) -> Iterator[tuple]:
    f_count = len(factories)
    if f_count == 0:
        yield initial_data
        return
    iterators = [None] * f_count  # type: List[Optional[Iterator[tuple]]]
    next_data = initial_data
    i = 0
    while True:
        try:
            while i < f_count:
                if iterators[i] is None:
                    iterators[i] = factories[i](*next_data)
                next_data = iterators[i].__next__()
                i += 1
            yield next_data
            i -= 1
        except StopIteration:
            iterators[i] = None
            i -= 1
            if i < 0:
                break


class cached_property(property):
    def __init__(self, getter):
        super().__init__(getter)
        self._name = getter.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        if self._name not in obj.__dict__:
            obj.__dict__[self._name] = self.fget(obj)
        return obj.__dict__[self._name]


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)

