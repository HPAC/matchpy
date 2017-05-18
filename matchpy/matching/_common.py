# -*- coding: utf-8 -*-
"""This module contains the CommutativePatternsParts class which is used by multiple matching algorithms."""
from typing import Callable, Dict, Iterator, NamedTuple, Optional, Sequence, Type, cast  # pylint: disable=unused-import

from multiset import Multiset

from ..expressions.expressions import Expression, Operation, Wildcard
from ..expressions.substitution import Substitution
from ..expressions.functions import is_constant, is_syntactic

__all__ = ['CommutativePatternsParts', 'Matcher', 'VarInfo']

Matcher = Callable[[Sequence[Expression], Expression, Substitution], Iterator[Substitution]]
VarInfo = NamedTuple('VarInfo', [('min_count', int), ('type', Optional[type])])


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

        constant (Multiset):
            A :class:`~.Multiset` representing the constant operands of the pattern.
            An expression is constant, if it does not contain variables or wildcards.
        syntactic (Multiset[Operation]):
            A :class:`.Multiset` representing the syntactic operands of the pattern.
            An expression is syntactic, if it does contain neither associative nor commutative operations
            nor sequence variables. Here, constant expressions and variables also get their own counters,
            so they are not included in this counter.
        sequence_variables (Multiset[str]):
            A :class:`.Multiset` representing the sequence variables of the pattern.
            Variables are represented by their name. Additional information is stored in
            ``sequence_variable_infos``. For wildcards without variable, the name will be ``None``.
        sequence_variable_infos (Dict[str, VarInfo]):
            A dictionary mapping sequence variable names to more information about the variable, i.e. its
            ``min_count`` and ``constraint``.
        fixed_variables (Multiset[VarInfo]):
            A :class:`.Multiset` representing the fixed length variables of the pattern.
            Here the key is a tuple of the form `(name, length)` of the variable.
            For wildcards without variable, the name will be `None`.
        fixed_variable_infos (Dict[str, VarInfo]):
            A dictionary mapping fixed variable names to more information about the variable, i.e. its
            ``min_count`` and ``constraint``.
        rest (Multiset):
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

        self.constant = Multiset()  # type: Multiset
        self.syntactic = Multiset()  # type: Multiset
        self.sequence_variables = Multiset()  # type: Multiset[str]
        self.sequence_variable_infos = dict()
        self.fixed_variables = Multiset()  # type: Multiset[str]
        self.fixed_variable_infos = dict()
        self.rest = Multiset()  # type: Multiset

        self.sequence_variable_min_length = 0
        self.fixed_variable_length = 0
        self.wildcard_min_length = 0
        self.wildcard_fixed = None

        for expression in expressions:
            expression = expression
            if is_constant(expression):
                self.constant[expression] += 1
            elif isinstance(expression, Wildcard):
                wc = cast(Wildcard, expression)
                if wc.variable_name:
                    name = wc.variable_name
                    if wc.fixed_size:
                        self.fixed_variables[name] += 1
                        symbol_type = getattr(wc, 'symbol_type', None)
                        self._update_var_info(self.fixed_variable_infos, name, wc.min_count, symbol_type)
                        self.fixed_variable_length += wc.min_count
                    else:
                        self.sequence_variables[name] += 1
                        self._update_var_info(self.sequence_variable_infos, name, wc.min_count)
                        self.sequence_variable_min_length += wc.min_count
                else:
                    self.wildcard_min_length += wc.min_count
                    if self.wildcard_fixed is None:
                        self.wildcard_fixed = wc.fixed_size
                    else:
                        self.wildcard_fixed = self.wildcard_fixed and wc.fixed_size
            elif is_syntactic(expression):
                self.syntactic[expression] += 1
            else:
                self.rest[expression] += 1

    @staticmethod
    def _update_var_info(infos, name, count, symbol_type=None):
        if name not in infos:
            infos[name] = VarInfo(count, symbol_type)
        else:
            existing_info = infos[name]
            assert existing_info.min_count == count
            assert existing_info.type == symbol_type

    def __str__(self):
        parts = []
        parts.extend(map(str, self.constant))
        parts.extend(map(str, self.syntactic))
        parts.extend(map(str, self.rest))

        for name, count in self.sequence_variables.items():
            parts.extend([name] * count)

        for name, count in self.fixed_variables.items():
            parts.extend([name] * count)

        return '{}({})'.format(self.operation.name, ', '.join(parts))
