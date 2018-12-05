# -*- coding: utf-8 -*-
"""This module contains various utility functions."""
import inspect
import itertools
import math
import ast
import os
import tokenize
import copy
from types import LambdaType

# pylint: disable=unused-import
from typing import (Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, TypeVar, cast, Union, Any)
# pylint: enable=unused-import

from multiset import Multiset

__all__ = [
    'fixed_integer_vector_iter', 'weak_composition_iter', 'commutative_sequence_variable_partition_iter',
    'get_short_lambda_source', 'solve_linear_diop', 'generator_chain', 'cached_property', 'slot_cached_property',
    'extended_euclid', 'base_solution_linear'
]

T = TypeVar('T')
VariableWithCount = NamedTuple(
    'VariableWithCount', [('name', str), ('count', int), ('minimum', int), ('default', Optional[Any])]
)


def fixed_integer_vector_iter(max_vector: Tuple[int, ...], vector_sum: int) -> Iterator[Tuple[int, ...]]:
    """
    Return an iterator over the integer vectors which

    - are componentwise less than or equal to *max_vector*, and
    - are non-negative, and where
    - the sum of their components is exactly *vector_sum*.

    The iterator yields the vectors in lexicographical order.

    Examples:

        List all vectors that are between ``(0, 0)`` and ``(2, 2)`` componentwise, where the sum of components is 2:

        >>> vectors = list(fixed_integer_vector_iter([2, 2], 2))
        >>> vectors
        [(0, 2), (1, 1), (2, 0)]
        >>> list(map(sum, vectors))
        [2, 2, 2]

    Args:
        max_vector:
            Maximum vector for the iteration. Every yielded result will be less than or equal to this componentwise.
        vector_sum:
            Every iterated vector will have a component sum equal to this value.

    Yields:
        All non-negative vectors that have the given sum and are not larger than the given maximum.

    Raises:
        ValueError:
            If *vector_sum* is negative.
    """
    if vector_sum < 0:
        raise ValueError("Vector sum must not be negative")
    if len(max_vector) == 0:
        if vector_sum == 0:
            yield tuple()
        return
    total = sum(max_vector)
    if vector_sum <= total:
        start = max(max_vector[0] + vector_sum - total, 0)
        end = min(max_vector[0], vector_sum)
        for j in range(start, end + 1):
            for vec in fixed_integer_vector_iter(max_vector[1:], vector_sum - j):
                yield (j, ) + vec


def weak_composition_iter(n: int, num_parts: int) -> Iterator[Tuple[int, ...]]:
    """Yield all weak compositions of integer *n* into *num_parts* parts.

    Each composition is yielded as a tuple. The generated partitions are order-dependant and not unique when
    ignoring the order of the components. The partitions are yielded in lexicographical order.

    Example:

        >>> compositions = list(weak_composition_iter(5, 2))
        >>> compositions
        [(0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0)]

        We can easily verify that all compositions are indeed valid:

        >>> list(map(sum, compositions))
        [5, 5, 5, 5, 5, 5]

    The algorithm was adapted from an answer to this `Stackoverflow question`_.

    Args:
        n:
            The integer to partition.
        num_parts:
            The number of parts for the combination.

    Yields:
        All non-negative tuples that have the given sum and size.

    Raises:
        ValueError:
            If *n* or *num_parts* are negative.

    .. _Stackoverflow question: http://stackoverflow.com/questions/40538923/40540014#40540014
    """
    if n < 0:
        raise ValueError("Total must not be negative")
    if num_parts < 0:
        raise ValueError("Number of num_parts must not be negative")
    if num_parts == 0:
        if n == 0:
            yield tuple()
        return
    m = n + num_parts - 1
    last = (m, )
    first = (-1, )
    for t in itertools.combinations(range(m), num_parts - 1):
        yield tuple(v - u - 1 for u, v in zip(first + t, t + last))


def optional_iter(remaining, count):
    for p in itertools.product([0, 1], repeat=count):
        yield remaining - sum(p), p


_linear_diop_solution_cache = {}  # type: Dict[Tuple[int, ...], List[Tuple[int, ...]]]


def _make_variable_generator_factory(value, total, variables: List[VariableWithCount]):
    var_counts = [v.count for v in variables]
    cache_key = (total, *var_counts)

    def _factory(subst):
        if cache_key in _linear_diop_solution_cache:
            solutions = _linear_diop_solution_cache[cache_key]
        else:
            solutions = list(solve_linear_diop(total, *var_counts))
            _linear_diop_solution_cache[cache_key] = solutions
        for solution in solutions:
            new_subst = copy.copy(subst)
            for var, count in zip(variables, solution):
                new_subst[var.name][value] = count
            yield new_subst

    return _factory


def _commutative_single_variable_partiton_iter(values: 'Multiset[T]',
                                               variable: VariableWithCount) -> Iterator[Dict[str, 'Multiset[T]']]:
    name, count, minimum, default = variable
    if len(values) == 0 and default is not None:
        yield {name: default}
        return
    if count == 1:
        if len(values) >= minimum:
            yield {name: values} if name is not None else {}
    else:
        new_values = Multiset()
        for element, element_count in values.items():
            if element_count % count != 0:
                return
            new_values[element] = element_count // count
        if len(new_values) >= minimum:
            yield {name: new_values} if name is not None else {}


def commutative_sequence_variable_partition_iter(values: Multiset, variables: List[VariableWithCount]
                                                ) -> Iterator[Dict[str, Multiset]]:
    """Yield all possible variable substitutions for given values and variables.

    .. note::

        The results are not yielded in any particular order because the algorithm uses dictionaries. Dictionaries until
        Python 3.6 do not keep track of the insertion order.

    Example:

        For a subject like ``fc(a, a, a, b, b, c)`` and a pattern like ``f(x__, y___, y___)`` one can define the
        following input parameters for the partitioning:

        >>> x = VariableWithCount(name='x', count=1, minimum=1, default=None)
        >>> y = VariableWithCount(name='y', count=2, minimum=0, default=None)
        >>> values = Multiset('aaabbc')

        Then the solutions are found (and sorted to get a unique output):

        >>> substitutions = commutative_sequence_variable_partition_iter(values, [x, y])
        >>> as_strings = list(str(Substitution(substitution)) for substitution in substitutions)
        >>> for substitution in sorted(as_strings):
        ...     print(substitution)
        {x ↦ {a, a, a, b, b, c}, y ↦ {}}
        {x ↦ {a, a, a, c}, y ↦ {b}}
        {x ↦ {a, b, b, c}, y ↦ {a}}
        {x ↦ {a, c}, y ↦ {a, b}}

    Args:
        values:
            The multiset of values which are partitioned and distributed among the variables.
        variables:
            A list of the variables to distribute the values among. Each variable has a name, a count of how many times
            it occurs and a minimum number of values it needs.

    Yields:
        Each possible substitutions that is a valid partitioning of the values among the variables.
    """
    if len(variables) == 1:
        yield from _commutative_single_variable_partiton_iter(values, variables[0])
        return

    generators = []
    for value, count in values.items():
        generators.append(_make_variable_generator_factory(value, count, variables))

    initial = dict((var.name, Multiset()) for var in variables)  # type: Dict[str, 'Multiset[T]']
    for subst in generator_chain(initial, *generators):
        valid = True
        for var in variables:
            if var.default is not None and len(subst[var.name]) == 0:
                subst[var.name] = var.default
            elif len(subst[var.name]) < var.minimum:
                valid = False
                break
        if valid:
            if None in subst:
                del subst[None]
            yield subst

class LambdaNodeVisitor(ast.NodeVisitor):
    def __init__(self, lines):
        self.lines = lines
        self.last_node = None
        self.lambdas = []

    def visit(self, node):
        super().visit(node)
        if self.last_node is not None:
            fake_node = ast.Pass(lineno=len(self.lines), col_offset=len(self.lines[-1]))
            self.generic_visit(fake_node)

    def visit_Lambda(self, node):
        self.generic_visit(node)
        self.last_node = node

    def generic_visit(self, node):
        if self.last_node is not None and hasattr(node, 'col_offset'):
            enode = ast.Expression(self.last_node)
            lambda_code = compile(enode, '<unused>', 'eval')
            lines = self.lines[self.last_node.lineno-1:node.lineno]
            lines[-1] = lines[-1][:node.col_offset]
            lines[0] = lines[0][self.last_node.col_offset:]
            lambda_body_text = ' '.join(l.rstrip(' \t\\').strip() for l in lines)
            while lambda_body_text:
                try:
                    code = compile(lambda_body_text, '<unused>', 'eval')
                    if len(code.co_code) == len(lambda_code.co_code):
                        break
                except SyntaxError:
                    pass
                lambda_body_text = lambda_body_text[:-1]
            self.lambdas.append((lambda_code, lambda_body_text.strip()))
            self.last_node = None
        super().generic_visit(node)

def get_short_lambda_source(lambda_func: LambdaType) -> Optional[str]:
    """Return the source of a (short) lambda function.
    If it's impossible to obtain, return ``None``.

    The source is returned without the ``lambda`` and signature parts:

    >>> get_short_lambda_source(lambda x, y: x < y)
    'x < y'

    This should work well for most lambda definitions, however for multi-line or highly nested lambdas,
    the source extraction might not succeed.

    Args:
        lambda_func:
            The lambda function.

    Returns:
        The source of the lambda function without its signature.
    """
    try:
        all_source_lines, lnum = inspect.findsource(lambda_func)
        source_lines, _ = inspect.getsourcelines(lambda_func)
    except (IOError, TypeError):
        return None
    all_source_lines = [l.rstrip('\r\n') for l in all_source_lines]
    block_end = lnum + len(source_lines)
    source_ast = None
    for i in range(lnum, -1, -1):
        try:
            block = all_source_lines[i:block_end]
            if block[0].startswith(' ') or block[0].startswith('\t'):
                block.insert(0, 'with 0:')
            source_ast = ast.parse(os.linesep.join(block))
        except (SyntaxError, tokenize.TokenError):
            pass
        else:
            break
    nv = LambdaNodeVisitor(block)
    nv.visit(source_ast)
    lambda_code = lambda_func.__code__
    for candidate_code, lambda_text in nv.lambdas:
        candidate_code = candidate_code.co_consts[0]
        # We don't check for direct equivalence since the flags can be different
        if (candidate_code.co_code == lambda_code.co_code and
            candidate_code.co_consts == lambda_code.co_consts and
            candidate_code.co_names == lambda_code.co_names and
            candidate_code.co_varnames == lambda_code.co_varnames and
            candidate_code.co_cellvars == lambda_code.co_cellvars and
            candidate_code.co_freevars == lambda_code.co_freevars):
            return lambda_text[lambda_text.index(':')+1:].strip()

    return None

def extended_euclid(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm that computes the Bézout coefficients as well as :math:`gcd(a, b)`

    Returns ``x, y, d`` where *x* and *y* are a solution to :math:`ax + by = d` and :math:`d = gcd(a, b)`.
    *x* and *y* are a minimal pair of Bézout's coefficients.

    See `Extended Euclidean algorithm <https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm>`_ or
    `Bézout's identity <https://en.wikipedia.org/wiki/B%C3%A9zout%27s_identity>`_ for more information.

    Example:

        Compute the Bézout coefficients and GCD of 42 and 12:

        >>> a, b = 42, 12
        >>> x, y, d = extended_euclid(a, b)
        >>> x, y, d
        (1, -3, 6)

        Verify the results:

        >>> import math
        >>> d == math.gcd(a, b)
        True
        >>> a * x + b * y == d
        True

    Args:
        a:
            The first integer.
        b:
            The second integer.

    Returns:
        A tuple with the Bézout coefficients and the greatest common divider of the arguments.
    """
    if b == 0:
        return (1, 0, a)

    x0, y0, d = extended_euclid(b, a % b)
    x, y = y0, x0 - (a // b) * y0

    return (x, y, d)


def base_solution_linear(a: int, b: int, c: int) -> Iterator[Tuple[int, int]]:
    r"""Yield solutions for a basic linear Diophantine equation of the form :math:`ax + by = c`.

    First, the equation is normalized by dividing :math:`a, b, c` by their gcd.
    Then, the extended Euclidean algorithm (:func:`extended_euclid`) is used to find a base solution :math:`(x_0, y_0)`.

    All non-negative solutions are generated by using that the general solution is :math:`(x_0 + b t, y_0 - a t)`.
    Because the base solution is one of the minimal pairs of Bézout's coefficients, for all non-negative solutions
    either :math:`t \geq 0` or :math:`t \leq 0` must hold. Also, all the non-negative solutions are consecutive with
    respect to :math:`t`.

    Hence, by adding or subtracting :math:`a` resp. :math:`b` from the base solution, all non-negative solutions can
    be efficiently generated.

    Args:
        a:
            The first coefficient of the equation.
        b:
            The second coefficient of the equation.
        c:
            The constant of the equation.

    Yields:
        Each non-negative integer solution of the equation as a tuple ``(x, y)``.

    Raises:
        ValueError:
            If any of the coefficients is not a positive integer.
    """
    if a <= 0 or b <= 0:
        raise ValueError('Coefficients a and b must be positive integers.')
    if c < 0:
        raise ValueError('Constant c must not be negative.')

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
    r"""Yield non-negative integer solutions of a linear Diophantine equation of the format
    :math:`c_1 x_1 + \dots + c_n x_n = total`.

    If there are at most two coefficients, :func:`base_solution_linear()` is used to find the solutions.
    Otherwise, the solutions are found recursively, by reducing the number of variables in each recursion:

    1. Compute :math:`d := gcd(c_2, \dots , c_n)`
    2. Solve :math:`c_1 x + d y = total`
    3. Recursively solve :math:`c_2 x_2 + \dots + c_n x_n = y` for each solution for :math:`y`
    4. Combine these solutions to form a solution for the whole equation

    Args:
        total:
            The constant of the equation.
        *coeffs:
            The coefficients :math:`c_i` of the equation.

    Yields:
        The non-negative integer solutions of the equation as a tuple :math:`(x_1, \dots, x_n)`.
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


def generator_chain(initial_data: T, *factories: Callable[[T], Iterator[T]]) -> Iterator[T]:
    """Chain multiple generators together by passing results from one to the next.

    This helper function allows to create a chain of generator where each generator is constructed by a factory that
    gets the data yielded by the previous generator. So each generator can generate new data dependant on the data
    yielded by the previous one. For each data item yielded by a generator, a new generator is constructed by the
    next factory.

    Example:

        Lets say for every number from 0 to 4, we want to count up to that number. Then we can do
        something like this using list comprehensions:

        >>> [i for n in range(1, 5) for i in range(1, n + 1)]
        [1, 1, 2, 1, 2, 3, 1, 2, 3, 4]

        You can use this function to achieve the same thing:

        >>> list(generator_chain(5, lambda n: iter(range(1, n)), lambda i: iter(range(1, i + 1))))
        [1, 1, 2, 1, 2, 3, 1, 2, 3, 4]

        The advantage is, that this is independent of the number of dependant generators you have.
        Also, this function does not use recursion so it is safe to use even with large generator counts.

    Args:
        initial_data:
            The initial data that is passed to the first generator factory.
        *factories:
            The generator factories. Each of them gets passed its predecessors data and has to return an iterable.
            The data from this iterable is passed to the next factory.

    Yields:
        Every data item yielded by the generators of the final factory.

    """
    generator_count = len(factories)
    if generator_count == 0:
        yield initial_data
        return
    generators = [None] * generator_count  # type: List[Optional[Iterator[T]]]
    next_data = initial_data
    generator_index = 0
    while True:
        try:
            while generator_index < generator_count:
                if generators[generator_index] is None:
                    generators[generator_index] = factories[generator_index](next_data)
                next_data = next(generators[generator_index])
                generator_index += 1
            yield next_data
            generator_index -= 1
        except StopIteration:
            generators[generator_index] = None
            generator_index -= 1
            if generator_index < 0:
                break


class cached_property(property):
    """Property with caching.

    An extension of the builtin `property`, that caches the value after the first access.
    This is useful in case the computation of the property value is expensive.

    Use it just like a regular property decorator. Cached properties cannot have a setter.

    Example:

        First, create a class with a cached property:

        >>> class MyClass:
        ...     @cached_property
        ...     def my_property(self):
        ...         print('my_property called')
        ...         return 42
        >>> instance = MyClass()

        Then, access the property and get the computed value:

        >>> instance.my_property
        my_property called
        42

        Now the result is cached and the original method will not be called again:

        >>> instance.my_property
        42
    """

    def __init__(self, getter, slot=None):
        """
        Use it as a decorator:

        >>> class MyClass:
        ...     @cached_property
        ...     def my_property(self):
        ...         return 42

        The *slot* argument specifies a class slot to use for caching the property. You should use the
        `slot_cached_property` decorator instead as that is more convenient.

        Args:
            getter:
                The getter method for the property.
            slot:
                Optional slot to use for the cached value. Only relevant in classes that use slots.
                Use `slot_cached_property` instead.

        Returns:
            The wrapped `property` with caching.
        """
        super().__init__(getter)
        self._name = getter.__name__
        self._slot = slot

    def __get__(self, obj, cls):
        if obj is None:
            return self
        if self._slot is not None:
            attribute = cls.__dict__[self._slot]
            try:
                return attribute.__get__(obj, cls)
            except AttributeError:
                value = self.fget(obj)
                attribute.__set__(obj, value)
                return value
        else:
            if self._name not in obj.__dict__:
                obj.__dict__[self._name] = self.fget(obj)
            return obj.__dict__[self._name]


def slot_cached_property(slot):
    """Property with caching for classes with slots.

    This is a wrapper around `cached_property` to be used with classes that have slots.
    It provides an extension of the builtin `property`, that caches the value in a slot after the first access.
    You need to specify which slot to use for the cached value.

    Example:

        First, create a class with a cached property and a slot to hold the cached value:

        >>> class MyClass:
        ...     __slots__ = ('_my_cached_property', )
        ...
        ...     @slot_cached_property('_my_cached_property')
        ...     def my_property(self):
        ...         print('my_property called')
        ...         return 42
        ...
        >>> instance = MyClass()

        Then, access the property and get the computed value:

        >>> instance.my_property
        my_property called
        42

        Now the result is cached and the original method will not be called again:

        >>> instance.my_property
        42

    Args:
        slot:
            The name of the classes slot to use for the cached value.

    Returns:
        The wrapped `cached_property`.
    """

    def _wrapper(getter):
        return cached_property(getter, slot)

    return _wrapper
