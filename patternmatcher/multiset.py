# -*- coding: utf-8 -*-
"""Contains the :class:`Multiset` class."""

from collections import Counter
from typing import Generic, Mapping, TypeVar

T = TypeVar('T')

class Multiset(Counter, Mapping[T, int], Generic[T]): # pylint: disable=abstract-method
    """A version of :class:`collections.Counter` that supports comparison."""

    def __le__(self, other: Counter):
        """Check if all counts from this counter are less than or equal to the other.

        >>> Multiset('ab') <= Multiset('aabc')
        True
        """
        if not isinstance(other, Counter):
            return NotImplemented
        for elem in self:
            if self[elem] > other[elem]:
                return False
        return True

    def __ge__(self, other: Counter):
        """Check if all counts from this counter are greater than or equal to the other.

        >>> Multiset('aabc') >= Multiset('ab')
        True
        """
        if not isinstance(other, Counter):
            return NotImplemented
        for elem in self:
            if self[elem] < other[elem]:
                return False
        return True
