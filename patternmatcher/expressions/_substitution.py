# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple, Union, cast

from multiset import Multiset

from ._base import Expression
from ._expressions import Operation, Variable

VariableReplacement = Union[Tuple[Expression, ...], Multiset[Expression], Expression]


class Substitution(Dict[str, VariableReplacement]):
    """Special :class:`dict` for substitutions with nicer formatting.

    The key is a variable's name and the value the substitution for it.
    """

    def try_add_variable(self, variable: str, replacement: VariableReplacement) -> None:
        """Try to add the variable with its replacement to the substitution.

        This considers an existing replacement and will only succeed if the new replacement
        can be merged with the old replacement. Merging can occur if either the two replacements
        are equivalent. Replacements can also be merged if the old replacement for the variable was
        unordered (i.e. a :class:`~.Multiset`) and the new one is an equivalent ordered version of it:

        >>> subst = Substitution({'x': Multiset(['a', 'b'])})
        >>> subst.try_add_variable('x', ('a', 'b'))
        >>> print(subst)
        {x ↦ (a, b)}

        Args:
            variable:
                The name of the variable to add.
            replacement:
                The replacement for the variable.

        Raises:
            ValueError:
                if the variable cannot be merged because it conflicts with the existing
                substitution for the variable.
        """
        if variable not in self:
            self[variable] = replacement.copy() if isinstance(replacement, Multiset) else replacement
        else:
            existing_value = self[variable]

            if isinstance(existing_value, tuple):
                if isinstance(replacement, Multiset):
                    if Multiset(existing_value) != replacement:
                        raise ValueError
                elif replacement != existing_value:
                    raise ValueError
            elif isinstance(existing_value, Multiset):
                compare_value = Multiset(isinstance(replacement, Expression) and [replacement] or replacement)
                if existing_value == compare_value:
                    if not isinstance(replacement, Multiset):
                        self[variable] = replacement
                else:
                    raise ValueError
            elif replacement != existing_value:
                raise ValueError

    def union_with_variable(self, variable: str, replacement: VariableReplacement) -> 'Substitution':
        """Try to create a new substitution with the given variable added.

        See :meth:`try_add_variable` for a version of this method that modifies the substitution
        in place.

        Args:
            variable:
                The name of the variable to add.
            replacement:
                The substitution for the variable.

        Returns:
            The new substitution with the variable added or merged.

        Raises:
            ValueError:
                if the variable cannot be merged because it conflicts with the existing
                substitution for the variable.
        """
        new_subst = Substitution(self)
        new_subst.try_add_variable(variable, replacement)
        return new_subst

    def extract_substitution(self, expression: Expression, pattern: Expression) -> bool:
        """Extract the variable substitution for the given pattern and expression.

        This assumes that expression and pattern already match when being considered as linear.
        Also, they both must be :term:`syntactic`, as sequence variables cannot be handled here.
        All that this method does is checking whether all the substitutions for the variables can be unified.
        Also, this method mutates the substitution and might even do so in case the unification fails.
        So, in case it returns ``False``, the substitution is invalid for the match.

        Args:
            expression:
                A :term:`syntactic` expression that matches the pattern.
            pattern:
                A :term:`syntactic` pattern that matches the expression.

        Returns:
            ``True`` iff the substitution could be extracted successfully.
        """
        if isinstance(pattern, Variable):
            try:
                self.try_add_variable(pattern.name, expression)
            except ValueError:
                return False
            return self.extract_substitution(expression, pattern.expression)
        elif isinstance(pattern, Operation):
            assert isinstance(expression, type(pattern))
            assert len(expression.operands) == len(pattern.operands)
            op_expression = cast(Operation, expression)
            for expr, patt in zip(op_expression.operands, pattern.operands):
                if not self.extract_substitution(expr, patt):
                    return False
        return True

    def union(self, *others: 'Substitution') -> 'Substitution':
        """Try to merge the substitutions.

        If a variable occurs in multiple substitutions, try to merge the replacements.
        See :meth:`union_with_variable` to see how replacements are merged.

        >>> subst1 = Substitution({'x': Multiset(['a', 'b'])})
        >>> subst2 = Substitution({'x': ('a', 'b'), 'y': ('c', )})
        >>> print(subst1.union(subst2))
        {x ↦ (a, b), y ↦ (c)}

        Args:
            others:
                The other substitutions to merge with this one.

        Returns:
            The new substitution with the other substitutions merged.

        Raises:
            ValueError:
                if a variable occurs in multiple substitutions but cannot be merged because the
                substitutions conflict.
        """
        new_subst = Substitution(self)
        for other in others:
            for variable, replacement in other.items():
                new_subst.try_add_variable(variable, replacement)
        return new_subst

    def rename(self, renaming):
        return Substitution((renaming.get(name, name), value) for name, value in self.items())

    @staticmethod
    def _match_value_repr_str(value: Union[List[Expression], Expression]) -> str:  # pragma: no cover
        if isinstance(value, (list, tuple)):
            return '({!s})'.format(', '.join(str(x) for x in value))
        return str(value)

    def __str__(self):
        return '{{{}}}'.format(
            ', '.join('{!s} ↦ {!s}'.format(k, self._match_value_repr_str(v)) for k, v in sorted(self.items()))
        )

    def __repr__(self):
        return '{{{}}}'.format(', '.join('{!r}: {!r}'.format(k, v) for k, v in sorted(self.items())))
