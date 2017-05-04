# -*- coding: utf-8 -*-
"""This module contains the CommutativePatternsParts class which is used by multiple matching algorithms."""
from typing import Callable, Dict, Iterator, NamedTuple, Optional, Sequence, Type, cast  # pylint: disable=unused-import

from multiset import Multiset

from ..expressions import Expression, Operation, Wildcard

__all__ = ['CommutativePatternsParts']


class CommutativePatternsParts(object):
    """Representation of the parts of a commutative pattern expression.

    This data structure contains all the operands of a commutative operation pattern.
    They are distinguished by how they need to be matched against an expression.

    All parts are represented by a :class:`.Multiset`, because the order of operands does not matter
    in a commutative operation.

    In addition, some lengths are precalculated during the initialization of this data structure
    so that they do not have to be recalculated later.

    This data structure is meant to be immutable, so do not change any of its attributes!

    Attributes:
        operation (Type[Operation]):
            The type of of the original pattern expression. Must be a subclass of
            :class:`.Operation`.

        constant (Multiset[Expression]):
            A :class:`~.Multiset` representing the constant operands of the pattern.
            An expression is constant, if it does not contain variables or wildcards.
        fixed_variables (Multiset[str]):
            A :class:`.Multiset` representing the fixed length variables of the pattern.
        fixed_variables_type (Dict[str, Optional[Type[Symbol]]]):
            A dictionary mapping fixed variable names to an optional symbol type constraint.
        sequence_variables (Multiset[str]):
            A :class:`.Multiset` representing the sequence variables of the pattern.
        sequence_variables_min (Dict[str, int]):
            A dictionary mapping sequence variable names to its ``min_count``.
        variable_terms (Multiset[Expression]):
            A :class:`.Multiset` representing the operands of the pattern that .
        fixed_terms (Multiset[Expression]):
            A :class:`.Multiset` representing the operands of the pattern that do not fall
            into one of the previous categories. That means it contains operation expressions, which
            are not syntactic.

        length (int):
            The total count of operands of the commutative operation pattern.
        sequence_variable_min_length (int):
            The total combined minimum length of all sequence variables in the commutative
            operation pattern. This is the sum of the `min_count` attributes of the sequence
            variables.
        fixed_variable_length (int):
            The total combined length of all fixed length variables in the commutative
            operation pattern. This is the sum of the `min_count` attributes of the
            variables.
        wildcard_fixed (Optional[bool]):
            Iff none of the operands is an unnamed wildcards, it is ``None``.
            Iff there are any unnamed sequence wildcards, it is ``True``.
            Otherwise, it is ``False``.
        wildcard_min_length (int):
            If :attr:`wildcard_fixed` is not ``None``, this is the total combined minimum length of all unnamed
            wildcards.
    """

    def __init__(self, operation: Type[Operation], *expressions: Expression) -> None:
        """Create a CommutativePatternsParts instance.

        Args:
            operation:
                The type of the commutative operation. Must be a subclass of :class:`.Operation` with
                :attr:`~.Operation.commutative` set to ``True``.
            *expressions:
                The operands of the commutative operation.
        """
        self.operation = operation
        self.length = len(expressions)

        self.constant = Multiset()  # type: Multiset[Expression]
        self.sequence_variables = Multiset()  # type: Multiset[str]
        self.sequence_variables_min = dict() # type: Dict[str, int]
        self.sequence_variables_wrap = dict() # type: Dict[str, bool]
        self.fixed_variables = Multiset()  # type: Multiset[str]
        self.fixed_variables_type = dict() # type: Dict[str, Optional[Type[Symbol]]]
        self.variable_terms = Multiset()  # type: Multiset[Expression]
        self.fixed_terms = Multiset()  # type: Multiset[Expression]

        self.sequence_variable_min_length = 0
        self.fixed_variable_length = 0
        self.wildcard_min_length = 0
        self.wildcard_fixed = None

        for expression in expressions:
            if expression.is_constant:
                self.constant[expression] += 1
            elif isinstance(expression, Operation):
                self.fixed_terms[expression] += 1
            elif isinstance(expression, Wildcard):
                varname = expression.variable_name
                symbol_type = getattr(expression, 'symbol_type', None)
                if expression.variable_name is None:
                    self.wildcard_min_length += expression.min_count
                    if self.wildcard_fixed is None:
                        self.wildcard_fixed = expression.fixed_size
                    else:
                        self.wildcard_fixed = self.wildcard_fixed and expression.fixed_size
                elif expression.fixed_size and not (operation and operation.associative and symbol_type is None):
                    self.fixed_variables[varname] += 1
                    self.fixed_variables_type[varname] = symbol_type
                    self.fixed_variable_length += 1
                else:
                    self.sequence_variables[varname] += 1
                    self.sequence_variables_min[varname] = expression.min_count
                    self.sequence_variables_wrap[varname] = expression.fixed_size
                    self.sequence_variable_min_length += expression.min_count
            else:
                self.variable_terms[expression] += 1

    def __str__(self):
        parts = []
        parts.extend(map(str, self.constant))
        parts.extend(map(str, self.variable_terms))
        parts.extend(map(str, self.fixed_terms))

        for name, count in self.sequence_variables.items():
            parts.extend([name] * count)

        for name, count in self.fixed_variables.items():
            parts.extend([name] * count)

        return '{}({})'.format(self.operation.name, ', '.join(parts))
