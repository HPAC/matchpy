# -*- coding: utf-8 -*-
import unittest
from ddt import ddt, data, unpack

from patternmatcher.multiset import Multiset

@ddt
class MultisetTest(unittest.TestCase):
    """Unit tests for :class:`Multiset`"""

    def test_missing(self):
        m = Multiset()
        self.assertEqual(m[object()], 0)

    @unpack
    @data(
        ('aab',     ['abc'],                list('aaabbc')),
        ('aab',     [''],                   list('aab')),
        ('aab',     [{'a': 2, 'b': 1}],     list('aaaabb')),
        ('aab',     [{}],                   list('aab')),
        ('aab',     [{'c': 0}],             list('aab')),
        ('a',       [Multiset('a')],        list('aa')),
        ('ab',      [Multiset()],           list('ab')),
        ('ab',      [],                     list('ab')),
        ('ab',      ['a', 'bc'],            list('aabbc')),
        ('ab',      [{}, ''],               list('ab')),
        ('ab',      [{'c': 1}, {'d': 0}],   list('abc')),
    )
    def test_update(self, initial, add, result):
        ms = Multiset(initial)       
        ms.update(*add)
        self.assertListEqual(sorted(ms), result)

if __name__ == '__main__':
    unittest.main()