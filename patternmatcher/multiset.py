# -*- coding: utf-8 -*-
"""Contains the :class:`Multiset` class."""

from collections.abc import MutableSet, Set
from typing import Generic, Iterable, Mapping, Optional, TypeVar, Union

from sortedcontainers import SortedDict

T = TypeVar('T')
OtherType = Union[Iterable[T], Mapping[T, int]]

class Multiset(dict, MutableSet, Mapping[T, int], Generic[T]):
    """A multiset implementation.

    A multiset is similar to the builtin :class:`set`, but elements can occur multiple times in the multiset.
    It is also similar to a :class:`list` without ordering of the values and hence no index-based operations.

    The multiset is implemented as a specialized :class:`dict` where the key is the element and the value its
    multiplicity. It supports all operations, that the :class:`set` supports

    In contrast to the builtin :class:`collections.Counter`, no negative counts are allowed, elements with
    zero counts are removed from the :class:`dict`, and set operations are supported.

    :see: https://en.wikipedia.org/wiki/Multiset
    """

    def __init__(self, iterable: Optional[OtherType]=None) -> None:
        r"""Create a new, empty Multiset object.

        And if given, initialize with elements from input iterable.
        Or, initialize from a mapping of elements to their multiplicity.

        Example:

        >>> ms = Multiset()                 # a new, empty multiset
        >>> ms = Multiset('abc')            # a new multiset from an iterable
        >>> ms = Multiset({'a': 4, 'b': 2}) # a new multiset from a mapping

        Parameters:
            iterable: An optional :class:`~typing.Iterable`\[~T] or
                :class:`~typing.Mapping`\[~T, :class:`int`] to initialize the multiset from.
        """
        self._total = 0
        super().__init__()
        if iterable is not None:
            self.update(iterable)

    def __missing__(self, element: T):
        """The multiplicity of elements not in the multiset is zero."""
        return 0

    def __setitem__(self, element: T, multiplicity: int):
        """Set the element's multiplicity.
        This will remove the element if the multiplicity is less than or equal to zero.
        '"""
        old = self[element]
        new = multiplicity > 0 and multiplicity or 0
        if multiplicity <= 0:
            if element in self:
                super().__delitem__(element)
        else:
            super().__setitem__(element, multiplicity)
        self._total += new - old

    def __str__(self):
        return '{%s}' % ', '.join(map(str, self))

    def __repr__(self):
        items = ', '.join('%r: %r' % item for item in self.items())
        return '%s({%s})' % (type(self).__name__, items)

    def __len__(self):
        """Returns the total number of elements in the multiset.

        Note that this is equivalent to the sum of the multiplicities:

        >>> ms = Multiset('aab')
        >>> len(ms)
        3
        >>> sum(ms.values())
        3

        If you need the total number of elements, use either the :meth:`keys`() method
        >>> len(ms.keys())
        2

        or convert to a :class:`set`:
        >>> len(set(ms))
        2
        """
        return self._total

    def __iter__(self):
        for element, multiplicity in self.items():
            for _ in range(multiplicity):
                yield element

    def update(self, *others: OtherType) -> None:
        r"""Like :meth:`dict.update` but add multiplicities instead of replacing them.

        >>> ms = Multiset('aab')
        >>> ms.update('abc')
        >>> sorted(ms)
        ['a', 'a', 'a', 'b', 'b', 'c']

        Note that the operator ``+=`` is equivalent to :meth:`update`, except that the operator will only
        accept sets to avoid accidental errors.

        >>> ms += Multiset('bc')
        >>> sorted(ms)
        ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c']

        For a variant of the operation which does not modify the multiset, but returns a new
        multiset instead see :meth:`combine`.

        Parameters:
            others: The other sets to add to this multiset. Can also be any :class:`~typing.Iterable`\[~T]
                or :class:`~typing.Mapping`\[~T, :class:`int`] which are then converted to :class:`Multiset`\[~T].
        """
        for other in others:
            if isinstance(other, Mapping):
                for elem, multiplicity in other.items():
                    self[elem] += multiplicity
            else:
                for elem in other:
                    self[elem] += 1

    def union_update(self, *others: OtherType) -> None:
        r"""Update the multiset, adding elements from all others using the maximum multiplicity.

        >>> ms = Multiset('aab')
        >>> ms.union_update('bc')
        >>> sorted(ms)
        ['a', 'a', 'b', 'c']

        You can also use the ``|=`` operator for the same effect. However, the operator version
        will only accept a set as other operator, not any iterable, to avoid errors.

        >>> ms = Multiset('aab')
        >>> ms |= Multiset('bccd')
        >>> sorted(ms)
        ['a', 'a', 'b', 'c', 'c', 'd']

        For a variant of the operation which does not modify the multiset, but returns a new
        multiset instead see :meth:`union`.

        Parameters:
            others: The other sets to union this multiset with. Can also be any :class:`~typing.Iterable`\[~T]
                or :class:`~typing.Mapping`\[~T, :class:`int`] which are then converted to :class:`Multiset`\[~T].
        """
        for other in map(self._as_multiset, others):
            for elem, multiplicity in other.items():
                if multiplicity > self[elem]:
                    self[elem] = multiplicity

    def __ior__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        self.union_update(other)
        return self

    def intersection_update(self, *others: OtherType) -> None:
        r"""Update the multiset, keeping only elements found in it and all others.

        >>> ms = Multiset('aab')
        >>> ms.intersection_update('bc')
        >>> sorted(ms)
        ['b']

        You can also use the ``&=`` operator for the same effect. However, the operator version
        will only accept a set as other operator, not any iterable, to avoid errors.

        >>> ms = Multiset('aabc')
        >>> ms &= Multiset('abbd')
        >>> sorted(ms)
        ['a', 'b']

        For a variant of the operation which does not modify the multiset, but returns a new
        multiset instead see :meth:`intersection`.

        Parameters:
            others: The other sets to intersect this multiset with. Can also be any :class:`~typing.Iterable`\[~T]
                or :class:`~typing.Mapping`\[~T, :class:`int`] which are then converted to :class:`Multiset`\[~T].
        """
        for other in map(self._as_multiset, others):
            for elem, current_count in list(self.items()):
                multiplicity = other[elem]
                if multiplicity < current_count:
                    self[elem] = multiplicity

    def __iand__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        self.intersection_update(other)
        return self

    def difference_update(self, *others: OtherType) -> None:
        r"""Remove all elements contained the others from this multiset.

        >>> ms = Multiset('aab')
        >>> ms.difference_update('abc')
        >>> sorted(ms)
        ['a']

        You can also use the ``-=`` operator for the same effect. However, the operator version
        will only accept a set as other operator, not any iterable, to avoid errors.

        >>> ms = Multiset('aabbbc')
        >>> ms -= Multiset('abd')
        >>> sorted(ms)
        ['a', 'b', 'b', 'c']

        For a variant of the operation which does not modify the multiset, but returns a new
        multiset instead see :meth:`difference`.

        Parameters:
            others: The other sets to remove from this multiset. Can also be any :class:`~typing.Iterable`\[~T]
                or :class:`~typing.Mapping`\[~T, :class:`int`] which are then converted to :class:`Multiset`\[~T].
        """
        for other in map(self._as_multiset, others):
            for elem, multiplicity in other.items():
                self.discard(elem, multiplicity)

    def __isub__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        self.difference_update(other)
        return self

    def symmetric_difference_update(self, other: OtherType) -> None:
        r"""Update the multiset to contain only elements in either this multiset or the other but not both.

        >>> ms = Multiset('aab')
        >>> ms.symmetric_difference_update('abc')
        >>> sorted(ms)
        ['a', 'c']

        You can also use the ``^=`` operator for the same effect. However, the operator version
        will only accept a set as other operator, not any iterable, to avoid errors.

        >>> ms = Multiset('aabbbc')
        >>> ms ^= Multiset('abd')
        >>> sorted(ms)
        ['a', 'b', 'b', 'c', 'd']

        For a variant of the operation which does not modify the multiset, but returns a new
        multiset instead see :meth:`symmetric_difference`.

        Parameters:
            other: The other set to take the symmetric difference with. Can also be any :class:`~typing.Iterable`\[~T]
                or :class:`~typing.Mapping`\[~T, :class:`int`] which are then converted to :class:`Multiset`\[~T].
        """
        other = self._as_multiset(other)
        keys = set(self.keys()) | set(other.keys())
        for elem in keys:
            multiplicity = self[elem]
            other_count = other[elem]
            self[elem] = multiplicity > other_count and multiplicity - other_count or other_count - multiplicity

    def __ixor__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        self.symmetric_difference_update(other)
        return self

    def times_update(self, factor: int) -> None:
        """Update each this multiset by multiplying each element's multiplicity with the given scalar factor.

        >>> ms = Multiset('aab')
        >>> ms.times_update(2)
        >>> sorted(ms)
        ['a', 'a', 'a', 'a', 'b', 'b']

        You can also use the ``*=`` operator for the same effect:

        >>> ms = Multiset('ac')
        >>> ms *= 3
        >>> sorted(ms)
        ['a', 'a', 'a', 'c', 'c', 'c']

        For a variant of the operation which does not modify the multiset, but returns a new
        multiset instead see :meth:`times`.

        Parameters:
            factor: The factor to multiply each multiplicity with.
        """
        if factor <= 0:
            self.clear()
        else:
            for elem in self.keys():
                self[elem] *= factor

    def __imul__(self, factor):
        self.times_update(factor)
        return self

    def add(self, element: T, multiplicity: int=1) -> None: # pylint: disable=arguments-differ
        """Adds an element to the multiset.

        >>> ms = Multiset()
        >>> ms.add('a')
        >>> sorted(ms)
        ['a']

        An optional multiplicity can be specified to define how many of the element are added:

        >>> ms.add('b', 2)
        >>> sorted(ms)
        ['a', 'b', 'b']

        This extends the :meth:`MutableSet.add` signature to allow specifying the multiplicity.

        Parameters:
            element:
                The element to add to the multiset.
            multiplicity:
                The multiplicity i.e. count of elements to add.
        """
        self[element] = self[element] + multiplicity

    def remove(self, element: T, multiplicity: Optional[int]=None) -> int: # pylint: disable=arguments-differ
        """Removes an element from the multiset.

        If no multiplicity is specified, the element is completely removed from the multiset:

        >>> ms = Multiset('aabbbc')
        >>> ms.remove('a')
        2
        >>> sorted(ms)
        ['b', 'b', 'b', 'c']

        If the multiplicity is given, it is subtracted from the element's multiplicity in the multiset:

        >>> ms.remove('b', 2)
        3
        >>> sorted(ms)
        ['b', 'c']

        This extends the :meth:`MutableSet.remove` signature to allow specifying the multiplicity.

        Parameters:
            element:
                The element to remove from the multiset.
            multiplicity:
                An optional multiplicity i.e. count of elements to remove.

        Returns:
            The multiplicity of the element in the multiset before
            the removal.

        Raises:
            KeyError: if the element is not contained in the set. Use :meth:`discard` if
                you do not want an exception to be raised.
        """
        if element not in self:
            raise KeyError
        old_count = self[element]
        if multiplicity is None:
            del self[element]
        else:
            self[element] = self[element] - multiplicity
        return old_count

    def discard(self, element: T, multiplicity: Optional[int]=None) -> int: # pylint: disable=arguments-differ
        """Removes the `element` from the multiset.

        If multiplicity is ``None``, all occurances of the element are removed:

        >>> ms = Multiset('aab')
        >>> ms.discard('a')
        2
        >>> sorted(ms)
        ['b']

        Otherwise, the multiplicity is subtracted from the one in the multiset:

        >>> ms = Multiset('aab')
        >>> ms.discard('a', 1)
        2
        >>> sorted(ms)
        ['a', 'b']

        In contrast to :meth:`remove`, this does not raise an error if the
        element is not in the multiset:

        >>> ms = Multiset('a')
        >>> ms.discard('b')
        0
        >>> sorted(ms)
        ['a']

        Parameters:
            element:
                The element to remove from the multiset.
            multiplicity:
                An optional multiplicity i.e. count of elements to remove.

        Returns:
            The multiplicity of the element in the multiset before
            the removal.
        """
        if element in self:
            old_count = self[element]
            if multiplicity is None:
                del self[element]
            else:
                self[element] -= multiplicity
            return old_count
        else:
            return 0

    def _as_multiset(self, other: OtherType) -> 'Multiset[T]':
        if not isinstance(other, Multiset):
            if not isinstance(other, Iterable):
                raise TypeError("'%s' object is not iterable" % type(other))
            return type(self)(other)
        return other

    def isdisjoint(self, other: OtherType) -> bool:
        r"""Return True if the set has no elements in common with other.

        Sets are disjoint iff their intersection is the empty set.

        >>> ms = Multiset('aab')
        >>> ms.isdisjoint('bc')
        False
        >>> ms.isdisjoint(Multiset('ccd'))
        True

        Parameters:
            other: The other set to check disjointness. Can also be an :class:`~typing.Iterable`\[~T]
                or :class:`~typing.Mapping`\[~T, :class:`int`] which are then converted to :class:`Multiset`\[~T].
        """
        other = self._as_multiset(other)
        for elem in self.keys():
            if elem in other:
                return False
        return True

    def difference(self, *others: OtherType) -> 'Multiset[T]':
        r"""Return a new multiset with all elements from the others removed.

        >>> ms = Multiset('aab')
        >>> sorted(ms.difference('bc'))
        ['a', 'a']

        You can also use the ``-`` operator for the same effect. However, the operator version
        will only accept a set as other operator, not any iterable, to avoid errors.

        >>> ms = Multiset('aabbbc')
        >>> sorted(ms - Multiset('abd'))
        ['a', 'b', 'b', 'c']

        For a variant of the operation which modifies the multiset in place see
        :meth:`difference_update`.

        Parameters:
            others: The other sets to remove from the multiset. Can also be any :class:`~typing.Iterable`\[~T]
                or :class:`~typing.Mapping`\[~T, :class:`int`] which are then converted to :class:`Multiset`\[~T].

        Returns:
            The resulting difference multiset.
        """
        result = type(self)(self)
        result.difference_update(*others)
        return result

    def __sub__(self, other: Set) -> bool:
        if not isinstance(other, Set):
            return NotImplemented
        return self.difference(other)

    def union(self, *others: OtherType) -> 'Multiset[T]':
        r"""Return a new multiset with all elements from the multiset and the others with maximal multiplicities.

        >>> ms = Multiset('aab')
        >>> sorted(ms.union('bc'))
        ['a', 'a', 'b', 'c']

        You can also use the ``|`` operator for the same effect. However, the operator version
        will only accept a set as other operator, not any iterable, to avoid errors.

        >>> ms = Multiset('aab')
        >>> sorted(ms | Multiset('aaa'))
        ['a', 'a', 'a', 'b']

        For a variant of the operation which modifies the multiset in place see
        :meth:`union`.

        Parameters:
            others: The other sets to union the multiset with. Can also be any :class:`~typing.Iterable`\[~T]
                or :class:`~typing.Mapping`\[~T, :class:`int`] which are then converted to :class:`Multiset`\[~T].

        Returns:
            The multiset resulting from the union.
        """
        result = type(self)(self)
        result.union_update(*others)
        return result

    def __or__(self, other: Set) -> bool:
        if not isinstance(other, Set):
            return NotImplemented
        return self.union(other)

    __ror__ = __or__

    def combine(self, *others: OtherType) -> 'Multiset[T]':
        r"""Return a new multiset with all elements from the multiset and the others with their multiplicities summed up.

        >>> ms = Multiset('aab')
        >>> sorted(ms.combine('bc'))
        ['a', 'a', 'b', 'b', 'c']

        You can also use the ``+`` operator for the same effect. However, the operator version
        will only accept a set as other operator, not any iterable, to avoid errors.

        >>> ms = Multiset('aab')
        >>> sorted(ms + Multiset('a'))
        ['a', 'a', 'a', 'b']

        For a variant of the operation which modifies the multiset in place see
        :meth:`update`.

        Parameters:
            others: The other sets to add to the multiset. Can also be any :class:`~typing.Iterable`\[~T]
                or :class:`~typing.Mapping`\[~T, :class:`int`] which are then converted to :class:`Multiset`\[~T].

        Returns:
            The multiset resulting from the addition of the sets.
        """
        result = type(self)(self)
        result.update(*others)
        return result

    def __add__(self, other: Set) -> bool:
        if not isinstance(other, Set):
            return NotImplemented
        return self.combine(other)

    __radd__ = __add__

    def intersection(self, *others: OtherType) -> 'Multiset[T]':
        r"""Return a new multiset with elements common to the multiset and all others.

        >>> ms = Multiset('aab')
        >>> sorted(ms.intersection('abc'))
        ['a', 'b']

        You can also use the ``&`` operator for the same effect. However, the operator version
        will only accept a set as other operator, not any iterable, to avoid errors.

        >>> ms = Multiset('aab')
        >>> sorted(ms & Multiset('aaac'))
        ['a', 'a']

        For a variant of the operation which modifies the multiset in place see
        :meth:`intersection_update`.

        Parameters:
            others: The other sets intersect with the multiset. Can also be any :class:`~typing.Iterable`\[~T]
                or :class:`~typing.Mapping`\[~T, :class:`int`] which are then converted to :class:`Multiset`\[~T].

        Returns:
            The multiset resulting from the intersection of the sets.
        """
        result = type(self)(self)
        result.intersection_update(*others)
        return result

    def __and__(self, other: Set) -> bool:
        if not isinstance(other, Set):
            return NotImplemented
        return self.intersection(other)

    __rand__ = __and__

    def symmetric_difference(self, other: OtherType) -> 'Multiset[T]':
        r"""Return a new set with elements in either the set or other but not both.

        >>> ms = Multiset('aab')
        >>> sorted(ms.symmetric_difference('abc'))
        ['a', 'c']

        You can also use the ``^`` operator for the same effect. However, the operator version
        will only accept a set as other operator, not any iterable, to avoid errors.

        >>> ms = Multiset('aab')
        >>> sorted(ms ^ Multiset('aaac'))
        ['a', 'b', 'c']

        For a variant of the operation which modifies the multiset in place see
        :meth:`symmetric_difference_update`.

        Parameters:
            other: The other set to take the symmetric difference with. Can also be any :class:`~typing.Iterable`\[~T]
                or :class:`~typing.Mapping`\[~T, :class:`int`] which are then converted to :class:`Multiset`\[~T].

        Returns:
            The resulting symmetric difference multiset.
        """
        result = type(self)(self)
        result.symmetric_difference_update(other)
        return result

    def __xor__(self, other: Set) -> bool:
        if not isinstance(other, Set):
            return NotImplemented
        return self.symmetric_difference(other)

    __rxor__ = __xor__

    def times(self, factor: int) -> 'Multiset[T]':
        """Return a new set with each element's multiplicity multiplied with the given scalar factor.

        >>> ms = Multiset('aab')
        >>> sorted(ms.times(2))
        ['a', 'a', 'a', 'a', 'b', 'b']

        You can also use the ``*`` operator for the same effect:

        >>> sorted(ms * 3)
        ['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b']

        For a variant of the operation which modifies the multiset in place see
        :meth:`times_update`.

        Parameters:
            factor: The factor to multiply each multiplicity with.
        """
        result = type(self)(self)
        result.times_update(factor)
        return result

    def __mul__(self, factor: int) -> 'Multiset[T]':
        if not isinstance(factor, int):
            return NotImplemented
        return self.times(factor)

    __rmul__ = __mul__

    def clear(self) -> None:
        """Empty the multiset."""
        super().clear()
        self._total = 0

    clear.__doc__ = dict.clear.__doc__

    def _issubset(self, other: OtherType, strict: bool) -> bool:
        other = self._as_multiset(other)
        other_len = len(other)
        if len(self) > other_len:
            return False
        if len(self) == other_len and strict:
            return False
        for elem, multiplicity in self.items():
            if multiplicity > other[elem]:
                return False
        return True

    def issubset(self, other: OtherType) -> bool:
        """Return True iff this set is a subset of the other.

        >>> Multiset('ab').issubset('aabc')
        True
        >>> Multiset('aabb').issubset(Multiset('aabc'))
        False

        You can also use the ``<=`` operator for this comparison:

        >>> Multiset('ab') <= Multiset('ab')
        True

        When using the ``<`` operator for comparison, the sets are checked
        to be unequal in addition:

        >>> Multiset('ab') < Multiset('ab')
        False

        Parameters:
            other: The potential superset of the multiset to be checked.

        Returns:
            True iff this set is a subset of the other.
        """
        return self._issubset(other, False)

    def __le__(self, other: Set) -> bool:
        if not isinstance(other, Set):
            return NotImplemented
        return self._issubset(other, False)

    def __lt__(self, other: Set) -> bool:
        if not isinstance(other, Set):
            return NotImplemented
        return self._issubset(other, True)

    def _issuperset(self, other: OtherType, strict: bool) -> bool:
        other = self._as_multiset(other)
        other_len = len(other)
        if len(self) < other_len:
            return False
        if len(self) == other_len and strict:
            return False
        for elem, multiplicity in other.items():
            if self[elem] < multiplicity:
                return False
        return True

    def issuperset(self, other: OtherType) -> bool:
        """Return True iff this multiset is a superset of the other.

        >>> Multiset('aabc').issuperset('ab')
        True
        >>> Multiset('aabc').issuperset(Multiset('abcc'))
        False

        You can also use the ``>=`` operator for this comparison:

        >>> Multiset('ab') >= Multiset('ab')
        True

        When using the ``>`` operator for comparison, the sets are checked
        to be unequal in addition:

        >>> Multiset('ab') > Multiset('ab')
        False

        Parameters:
            other: The potential subset of the multiset to be checked.

        Returns:
            True iff this set is a subset of the other.
        """
        return self._issuperset(other, False)

    def __ge__(self, other: Set) -> bool:
        if not isinstance(other, Set):
            return NotImplemented
        return self._issuperset(other, False)

    def __gt__(self, other: Set) -> bool:
        if not isinstance(other, Set):
            return NotImplemented
        return self._issuperset(other, True)

    def __eq__(self, other: Set):
        if not isinstance(other, Set):
            return NotImplemented
        if isinstance(other, Multiset):
            return dict.__eq__(self, other)
        if len(self) != len(other):
            return False
        return self._issubset(other, False)

    def copy(self):
        """Return a shallow copy of the multiset."""
        return type(self)(self)

    __copy__ = copy


class SortedMultiset(Multiset[T], SortedDict, Generic[T]):
    """A sorted variant of the :class:`Multiset`."""


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
