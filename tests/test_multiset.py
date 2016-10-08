# -*- coding: utf-8 -*-
import unittest
import doctest
from ddt import ddt, data, unpack

from patternmatcher.multiset import Multiset, SortedMultiset
import patternmatcher.multiset as multiset

@ddt
class MultisetTest(unittest.TestCase):
    """Unit tests for :class:`Multiset`"""

    def test_missing(self):
        m = Multiset()
        self.assertEqual(m[object()], 0)

    def test_setitem(self):
        m = Multiset()
        m[1] = 2
        self.assertEqual(m[1], 2)
        self.assertIn(1, m)

        m[1] = 0
        self.assertEqual(m[1], 0)
        self.assertNotIn(1, m)

        self.assertNotIn(2, m)
        m[2] = 0
        self.assertEqual(m[2], 0)
        self.assertNotIn(2, m)

        m[3] = -1
        self.assertEqual(m[3], 0)
        self.assertNotIn(3, m)

    def test_len(self):
        m = Multiset()
        self.assertEqual(len(m), 0)

        m.update('abc')
        self.assertEqual(len(m), 3)

        m.update('aa')
        self.assertEqual(len(m), 5)

        m['a'] = 1
        self.assertEqual(len(m), 3)

        m['b'] = 0
        self.assertEqual(len(m), 2)

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

    @unpack
    @data(
        ('aab',     ['abc'],                list('aabc')),
        ('aab',     [''],                   list('aab')),
        ('aab',     [{'a': 2, 'b': 1}],     list('aab')),
        ('aab',     [{}],                   list('aab')),
        ('aab',     [{'c': 0}],             list('aab')),
        ('a',       [Multiset('a')],        list('a')),
        ('ab',      [Multiset()],           list('ab')),
        ('ab',      [],                     list('ab')),
        ('ab',      ['a', 'bc'],            list('abc')),
        ('ab',      [{}, ''],               list('ab')),
        ('ab',      [{'c': 1}, {'d': 0}],   list('abc')),
        ('ab',      ['aa'],                 list('aab')),
    )
    def test_union_update(self, initial, add, result):
        ms = Multiset(initial)
        ms.union_update(*add)
        self.assertListEqual(sorted(ms), result)

    def test_union_update_error(self):
        with self.assertRaises(TypeError):
            Multiset().union_update(None)

    def test_ior(self):
        m = Multiset('ab')

        with self.assertRaises(TypeError):
            m |= 'abc'

        m |= Multiset('abc')
        self.assertEqual(sorted(m), list('abc'))

    @unpack
    @data(
        ('aab',     ['abc'],                list('ab')),
        ('aab',     [''],                   list()),
        ('aab',     [{'a': 2, 'b': 1}],     list('aab')),
        ('aab',     [{}],                   list()),
        ('aab',     [{'c': 0}],             list()),
        ('a',       [Multiset('a')],        list('a')),
        ('ab',      [Multiset()],           list()),
        ('ab',      [],                     list('ab')),
        ('ab',      ['a', 'bc'],            list()),
        ('ab',      ['a', 'aab'],           list('a')),
        ('ab',      [{}, ''],               list()),
        ('ab',      [{'c': 1}, {'d': 0}],   list()),
        ('ab',      ['aa'],                 list('a')),
    )
    def test_intersection_update(self, initial, args, result):
        ms = Multiset(initial)
        ms.intersection_update(*args)
        self.assertListEqual(sorted(ms), result)

    def test_iand(self):
        m = Multiset('aabd')

        with self.assertRaises(TypeError):
            m &= 'abc'

        m &= Multiset('abc')
        self.assertEqual(sorted(m), list('ab'))

    @unpack
    @data(
        ('aab',     'bc',                 list('aa')),
        ('aab',     'cd',                 list('aab')),
        ('aab',     'a',                  list('ab')),
        ('aab',     'aa',                 list('b')),
        ('aab',     '',                   list('aab')),
        ('aab',     {'a': 2, 'b': 1},     list()),
        ('aab',     {},                   list('aab')),
        ('aab',     {'c': 0},             list('aab')),
        ('a',       Multiset('a'),        list()),
        ('ab',      Multiset(),           list('ab')),
        ('ab',      'aa',                 list('b')),
    )
    def test_difference_update(self, initial, other, result):
        ms = Multiset(initial)
        ms.difference_update(other)
        self.assertListEqual(sorted(ms), result)

    def test_isub(self):
        m = Multiset('aabd')

        with self.assertRaises(TypeError):
            m -= 'abc'

        m -= Multiset('abc')
        self.assertEqual(sorted(m), list('ad'))

    @unpack
    @data(
        ('aab',     'bc',                 list('aac')),
        ('aab',     'cd',                 list('aabcd')),
        ('aab',     'a',                  list('ab')),
        ('aab',     'aa',                 list('b')),
        ('aab',     '',                   list('aab')),
        ('aab',     {'a': 2, 'b': 1},     list()),
        ('aab',     {},                   list('aab')),
        ('aab',     {'c': 0},             list('aab')),
        ('a',       Multiset('a'),        list()),
        ('ab',      Multiset(),           list('ab')),
        ('ab',      'aa',                 list('ab')),
    )
    def test_symmetric_difference_update(self, initial, other, result):
        ms = Multiset(initial)
        ms.symmetric_difference_update(other)
        self.assertListEqual(sorted(ms), result)

    def test_ixor(self):
        m = Multiset('aabd')

        with self.assertRaises(TypeError):
            m ^= 'abc'

        m ^= Multiset('abc')
        self.assertEqual(sorted(m), list('acd'))

    @unpack
    @data(
        ('aab',     2,                    list('aaaabb')),
        ('a',       3,                    list('aaa')),
        ('abc',     0,                    list()),
        ('abc',     1,                    list('abc'))
    )
    def test_times_update(self, initial, factor, result):
        ms = Multiset(initial)
        ms.times_update(factor)
        self.assertListEqual(sorted(ms), result)

    def test_imul(self):
        m = Multiset('aab')

        with self.assertRaises(TypeError):
            m *= 'a'

        m *= 2
        self.assertEqual(sorted(m), list('aaaabb'))

    def test_add(self):
        m = Multiset('aab')

        self.assertNotIn('c', m)
        m.add('c')
        self.assertIn('c', m)
        self.assertEqual(m['c'], 1)

        self.assertNotIn('d', m)
        m.add('d', 42)
        self.assertIn('d', m)
        self.assertEqual(m['d'], 42)

        m.add('c', 2)
        self.assertEqual(m['c'], 3)

    def test_remove(self):
        m = Multiset('aaaabbc')

        with self.assertRaises(KeyError):
            m.remove('x')

        self.assertIn('c', m)
        count = m.remove('c')
        self.assertNotIn('c', m)
        self.assertEqual(count, 1)

        self.assertIn('b', m)
        count = m.remove('b')
        self.assertNotIn('b', m)
        self.assertEqual(count, 2)

        self.assertIn('a', m)
        count = m.remove('a', 1)
        self.assertIn('a', m)
        self.assertEqual(count, 4)
        self.assertEqual(m['a'], 3)

        count = m.remove('a', 2)
        self.assertIn('a', m)
        self.assertEqual(count, 3)
        self.assertEqual(m['a'], 1)

        count = m.remove('a', 0)
        self.assertIn('a', m)
        self.assertEqual(count, 1)
        self.assertEqual(m['a'], 1)

    def test_discard(self):
        m = Multiset('aaaabbc')

        self.assertIn('c', m)
        count = m.discard('c')
        self.assertNotIn('c', m)
        self.assertEqual(count, 1)

        self.assertIn('b', m)
        count = m.discard('b')
        self.assertNotIn('b', m)
        self.assertEqual(count, 2)

        self.assertIn('a', m)
        count = m.discard('a', 1)
        self.assertIn('a', m)
        self.assertEqual(count, 4)
        self.assertEqual(m['a'], 3)

        count = m.discard('a', 2)
        self.assertIn('a', m)
        self.assertEqual(count, 3)
        self.assertEqual(m['a'], 1)

        count = m.discard('a', 0)
        self.assertIn('a', m)
        self.assertEqual(count, 1)
        self.assertEqual(m['a'], 1)

    @unpack
    @data(
        ('aab', 'a', False),
        ('aab', 'ab', False),
        ('a', 'aab', False),
        ('ab', 'aab', False),
        ('aab', 'c', True),
        ('aab', '', True),
        ('', 'abc', True),
    )
    def test_isdisjoint(self, set1, set2, disjoint):
        ms = Multiset(set1)
        if disjoint:
            self.assertTrue(ms.isdisjoint(set2))
        else:
            self.assertFalse(ms.isdisjoint(set2))


    @unpack
    @data(
        ('aab',     ['abc'],                list('aabc')),
        ('aab',     [''],                   list('aab')),
        ('aab',     [{'a': 2, 'b': 1}],     list('aab')),
        ('aab',     [{}],                   list('aab')),
        ('aab',     [{'c': 0}],             list('aab')),
        ('a',       [Multiset('a')],        list('a')),
        ('ab',      [Multiset()],           list('ab')),
        ('ab',      [],                     list('ab')),
        ('ab',      ['a', 'bc'],            list('abc')),
        ('ab',      [{}, ''],               list('ab')),
        ('ab',      [{'c': 1}, {'d': 0}],   list('abc')),
        ('ab',      ['aa'],                 list('aab')),
    )
    def test_union(self, initial, args, expected):
        ms = Multiset(initial)
        result = ms.union(*args)
        self.assertListEqual(sorted(result), expected)

    def test_union_error(self):
        with self.assertRaises(TypeError):
            Multiset().union(None)

    def test_or(self):
        m = Multiset('ab')

        with self.assertRaises(TypeError):
            _ = m | 'abc'

        result = m | Multiset('abc')
        self.assertEqual(sorted(result), list('abc'))

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
    def test_combine(self, initial, args, expected):
        ms = Multiset(initial)
        result = ms.combine(*args)
        self.assertListEqual(sorted(result), expected)

    def test_add_op(self):
        m = Multiset('aab')

        with self.assertRaises(TypeError):
            _ = m + 'abc'

        result = m + Multiset('abc')
        self.assertEqual(sorted(result), list('aaabbc'))

    @unpack
    @data(
        ('aab',     ['abc'],                list('ab')),
        ('aab',     [''],                   list()),
        ('aab',     [{'a': 2, 'b': 1}],     list('aab')),
        ('aab',     [{}],                   list()),
        ('aab',     [{'c': 0}],             list()),
        ('a',       [Multiset('a')],        list('a')),
        ('ab',      [Multiset()],           list()),
        ('ab',      [],                     list('ab')),
        ('ab',      ['a', 'bc'],            list()),
        ('ab',      ['a', 'aab'],           list('a')),
        ('ab',      [{}, ''],               list()),
        ('ab',      [{'c': 1}, {'d': 0}],   list()),
        ('ab',      ['aa'],                 list('a')),
    )
    def test_intersection(self, initial, args, expected):
        ms = Multiset(initial)
        result = ms.intersection(*args)
        self.assertListEqual(sorted(result), expected)

    def test_and(self):
        m = Multiset('aabd')

        with self.assertRaises(TypeError):
            _ = m & 'abc'

        result = m & Multiset('abc')
        self.assertEqual(sorted(result), list('ab'))

    @unpack
    @data(
        ('aab',     'bc',                 list('aa')),
        ('aab',     'cd',                 list('aab')),
        ('aab',     'a',                  list('ab')),
        ('aab',     'aa',                 list('b')),
        ('aab',     '',                   list('aab')),
        ('aab',     {'a': 2, 'b': 1},     list()),
        ('aab',     {},                   list('aab')),
        ('aab',     {'c': 0},             list('aab')),
        ('a',       Multiset('a'),        list()),
        ('ab',      Multiset(),           list('ab')),
        ('ab',      'aa',                 list('b')),
    )
    def test_difference(self, initial, other, expected):
        ms = Multiset(initial)
        result = ms.difference(other)
        self.assertListEqual(sorted(result), expected)

    def test_sub(self):
        m = Multiset('aabd')

        with self.assertRaises(TypeError):
            _ = m - 'abc'

        result = m - Multiset('abc')
        self.assertEqual(sorted(result), list('ad'))

    @unpack
    @data(
        ('aab',     'bc',                 list('aac')),
        ('aab',     'cd',                 list('aabcd')),
        ('aab',     'a',                  list('ab')),
        ('aab',     'aa',                 list('b')),
        ('aab',     '',                   list('aab')),
        ('aab',     {'a': 2, 'b': 1},     list()),
        ('aab',     {},                   list('aab')),
        ('aab',     {'c': 0},             list('aab')),
        ('a',       Multiset('a'),        list()),
        ('ab',      Multiset(),           list('ab')),
        ('ab',      'aa',                 list('ab')),
    )
    def test_symmetric_difference(self, initial, other, expected):
        ms = Multiset(initial)
        result = ms.symmetric_difference(other)
        self.assertListEqual(sorted(result), expected)

    def test_xor(self):
        m = Multiset('aabd')

        with self.assertRaises(TypeError):
            _ = m ^ 'abc'

        result = m ^ Multiset('abc')
        self.assertEqual(sorted(result), list('acd'))

    @unpack
    @data(
        ('aab',     2,                    list('aaaabb')),
        ('a',       3,                    list('aaa')),
        ('abc',     0,                    list()),
        ('abc',     1,                    list('abc'))
    )
    def test_times(self, initial, factor, expected):
        ms = Multiset(initial)
        result = ms.times(factor)
        self.assertListEqual(sorted(result), expected)

    def test_mul(self):
        m = Multiset('aab')

        with self.assertRaises(TypeError):
            _ = m * 'a'

        result = m * 2
        self.assertEqual(sorted(result), list('aaaabb'))

    @unpack
    @data(
        ('a',       'abc',  True),
        ('abc',     'abc',  True),
        ('',        'abc',  True),
        ('d',       'abc',  False),
        ('abcd',    'abc',  False),
        ('aabc',    'abc',  False),
        ('abd',     'abc',  False),
        ('a',    '',        False),
        ('',    'a',        True)
    )
    def test_issubset(self, set1, set2, issubset):
        ms = Multiset(set1)
        if issubset:
            self.assertTrue(ms.issubset(set2))
        else:
            self.assertFalse(ms.issubset(set2))

    def test_le(self):
        set1 = Multiset('ab')
        set2 = Multiset('aab')
        set3 = Multiset('ac')

        with self.assertRaises(TypeError):
            _ = set1 <= 'x'

        self.assertTrue(set1 <= set2)
        self.assertFalse(set2 <= set1)
        self.assertTrue(set1 <= set1)
        self.assertTrue(set2 <= set2)
        self.assertFalse(set1 <= set3)
        self.assertFalse(set3 <= set1)

    def test_lt(self):
        set1 = Multiset('ab')
        set2 = Multiset('aab')
        set3 = Multiset('ac')

        with self.assertRaises(TypeError):
            _ = set1 < 'x'

        self.assertTrue(set1 < set2)
        self.assertFalse(set2 < set1)
        self.assertFalse(set1 < set1)
        self.assertFalse(set2 < set2)
        self.assertFalse(set1 <= set3)
        self.assertFalse(set3 <= set1)

    @unpack
    @data(
        ('abc',  'a',       True),
        ('abc',  'abc',     True),
        ('abc',  '',        True),
        ('abc',  'd',       False),
        ('abc',  'abcd',    False),
        ('abc',  'aabc',    False),
        ('abc',  'abd',     False),
        ('a',    '',        True),
        ('',    'a',        False)
    )
    def test_issuperset(self, set1, set2, issubset):
        ms = Multiset(set1)
        if issubset:
            self.assertTrue(ms.issuperset(set2))
        else:
            self.assertFalse(ms.issuperset(set2))

    def test_ge(self):
        set1 = Multiset('ab')
        set2 = Multiset('aab')
        set3 = Multiset('ac')

        with self.assertRaises(TypeError):
            _ = set1 >= 'x'

        self.assertTrue(set1  >= set1)
        self.assertFalse(set1 >= set2)
        self.assertFalse(set1 >= set3)

        self.assertTrue(set2  >= set1)
        self.assertTrue(set2  >= set2)
        self.assertFalse(set2 >= set3)

        self.assertFalse(set3 >= set1)
        self.assertFalse(set3 >= set2)
        self.assertTrue(set3  >= set3)

    def test_gt(self):
        set1 = Multiset('ab')
        set2 = Multiset('aab')
        set3 = Multiset('ac')

        with self.assertRaises(TypeError):
            _ = set1 > 'x'

        self.assertFalse(set1 > set1)
        self.assertFalse(set1 > set2)
        self.assertFalse(set1 > set3)

        self.assertTrue(set2  > set1)
        self.assertFalse(set2 > set2)
        self.assertFalse(set2 > set3)

        self.assertFalse(set3 > set1)
        self.assertFalse(set3 > set2)
        self.assertFalse(set3 > set3)

    def test_compare_with_set(self):
        self.assertLessEqual(Multiset('ab'), set('ab'))
        self.assertLessEqual(Multiset('b'), set('ab'))
        self.assertGreaterEqual(Multiset('ab'), set('ab'))
        self.assertGreaterEqual(Multiset('abb'), set('ab'))
        self.assertLessEqual(set('ab'), Multiset('abb'))
        self.assertLessEqual(set('b'), Multiset('aab'))
        self.assertFalse(set('ab') >= Multiset('aab'))
        self.assertLessEqual(set('ab'), Multiset('aab'))
        self.assertGreaterEqual(set('ab'), Multiset('ab'))
    
    def test_eq_set(self):
        multisets = ['', 'a', 'ab', 'aa']
        sets = ['', 'a', 'ab']

        for i, ms in enumerate(multisets):
            ms = Multiset(ms)
            for j, s in enumerate(sets):
                s = set(s)
                if i == j:
                    self.assertTrue(ms == s)
                    self.assertTrue(s == ms)
                else:
                    self.assertFalse(ms == s)
                    self.assertFalse(s == ms)

    def test_eq(self):
        self.assertFalse(Multiset('ab') == Multiset('b'))
        self.assertFalse(Multiset('ab') == Multiset('a'))
        self.assertTrue(Multiset('ab') == Multiset('ab'))
        self.assertTrue(Multiset('aab') == Multiset('aab'))
        self.assertFalse(Multiset('aab') == Multiset('abb'))
        self.assertFalse(Multiset('ab') == 'ab')

    def test_ne_set(self):
        multisets = ['', 'a', 'ab', 'aa']
        sets = ['', 'a', 'ab']

        for i, ms in enumerate(multisets):
            ms = Multiset(ms)
            for j, s in enumerate(sets):
                s = set(s)
                if i == j:
                    self.assertFalse(ms != s)
                    self.assertFalse(s != ms)
                else:
                    self.assertTrue(ms != s)
                    self.assertTrue(s != ms)

    def test_ne(self):
        self.assertTrue(Multiset('ab') != Multiset('b'))
        self.assertTrue(Multiset('ab') != Multiset('a'))
        self.assertFalse(Multiset('ab') != Multiset('ab'))
        self.assertFalse(Multiset('aab') != Multiset('aab'))
        self.assertTrue(Multiset('aab') != Multiset('abb'))
        self.assertTrue(Multiset('ab') != 'ab')

    def test_copy(self):
        ms = Multiset('abc')

        ms_copy = ms.copy()

        self.assertEqual(ms, ms_copy)
        self.assertIsNot(ms, ms_copy)


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(multiset))
    return tests

if __name__ == '__main__':
    unittest.main()