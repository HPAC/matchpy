# -*- coding: utf-8 -*-
from typing import TypeVar, Tuple, List, Sequence, Iterator
import itertools
import ast
import math
import inspect
import re
from collections import Counter as Multiset

T = TypeVar('T')

def partitions_with_limits(values : List[T], limits : List[Tuple[int, int]]) -> Tuple[List[T], ...]:
    limits = list(limits)
    count = len(values)
    varCount = count - sum([m for (m, _) in limits])
    counts = []
    
    for (minCount, maxCount) in limits:
        if maxCount > minCount + varCount:
            maxCount = minCount + varCount
        counts.append(list(range(minCount, maxCount + 1)))
                        
    for countPartition in itertools.product(*counts):
        if sum(countPartition) != count:
            continue
            
        v = []
        i = 0
        
        for c in countPartition:
            v.append(values[i:i+c])
            i += c
            
        yield tuple(v)   

def partitions_with_count(n, m):
    # H1: Initialize
    a = [1] * m
    a.append(-1) 
    a[0] = n - m + 1

    while True:
        while True:
            # H2: Visit
            yield a[:-1]

            if a[1] > a[0] - 1:
                break

            # H3: Tweak a[0] and a[1]
            a[0] -= 1
            a[1] += 1

        # H4: Find j
        j = 2
        s = a[0] + a[1] - 1

        while a[j] >= a[0] - 1:
            s += a[j]
            j += 1

        # H5: Increase a[j]
        if j >= m:
            return
        
        x = a[j] + 1
        a[j] = x
        j -= 1

        # H6: Tweak a[:j]
        while j > 0:
            a[j] = x
            s -= x
            j -= 1
        
        a[0] = s

def fixed_sum_vector_iter(min_vect : Sequence[int], max_vect : Sequence[int], total : int) -> Iterator[List[int]]:
    assert len(min_vect) == len(max_vect), 'len(min_vect) != len(max_vect)'
    assert all(minValue <= maxValue for minValue, maxValue in zip(min_vect, max_vect)), 'min_vect > max_vect'

    minSum = sum(min_vect)
    maxSum = sum(max_vect)

    if minSum > total or maxSum < total:
        return

    count = len(max_vect)

    if count <= 1:
        yield len(max_vect) == 1 and [total] or []
        return

    remaining = total - minSum

    realMins = list(min_vect)
    realMaxs = list(max_vect)

    for i, (minimum, maximum) in enumerate(zip(min_vect, max_vect)):
        left_over_sum = sum(max_vect[:i] + max_vect[i+1:])
        if left_over_sum != math.inf:
            realMins[i] = max(total - left_over_sum, minimum)
        realMaxs[i] = min(remaining + minimum, maximum)

    values = list(realMins)

    remaining = total - sum(realMins)

    if remaining == 0:
        yield values
        return

    j = count - 1
    while remaining > 0:
        toAdd = min(realMaxs[j] - realMins[j], remaining)
        values[j] += toAdd
        remaining -= toAdd
        j -= 1

    while True:
        pos = count - 2
        yield values[:]
        while True:
            values[pos] += 1
            values[-1] -= 1
            if values[-1] < realMins[-1] or values[pos] > realMaxs[pos]:
                if pos == 0:
                    return
                variable_amount = values[pos] - realMins[pos] 
                values[pos] = realMins[pos] # reset current position
                values[-1] += variable_amount # reset last position

                if values[-1] > realMaxs[-1]:
                    remaining = values[-1] - realMaxs[-1] - 1
                    values[-1] = realMaxs[-1] + 1
                    j = count - 2
                    while remaining > 0:
                        toAdd = min(realMaxs[j] - values[j], remaining)
                        values[j] += toAdd
                        remaining -= toAdd
                        j -= 1
                pos -= 1
            else:
                break

def commutative_partition_iter(values: Sequence[T], min_vect: Sequence[int], max_vect: Sequence[int]) -> Iterator[Tuple[List[T], ...]]:
    counts = list(Multiset(values).items())
    counts.sort()
    value_count = len(counts)
    iterators = [None] * value_count
    values = [None] * value_count
    new_min = tuple(min_count == max_count and min_count or 0 for min_count, max_count in zip(min_vect, max_vect))
    iterators[0] = fixed_sum_vector_iter(new_min, max_vect, counts[0][1])
    try:
        values[0] = iterators[0].__next__()
    except IndexError:
        return
    i = 1
    while True:
        try:
            while i < value_count:
                if iterators[i] is None:
                    #other_counts = tuple(map(sum, zip(*values[:i])))
                    #new_min = tuple(max(min_count - s, 0) for min_count, s in zip(min_vect, other_counts))
                    #new_max = tuple(max(max_count - s, 0) for max_count, s in zip(max_vect, other_counts))
                    iterators[i] = fixed_sum_vector_iter(new_min, max_vect, counts[i][1])
                values[i] = iterators[i].__next__()
                i += 1
            sums = tuple(map(sum, zip(*values)))
            if all(minc <= s and s <= maxc for minc, s, maxc in zip(min_vect, sums, max_vect)):
                partiton = tuple([] for _ in range(len(min_vect)))
                for cs, (v, _) in zip(values, counts):
                    for j, c in enumerate(cs):
                        partiton[j].extend([v] * c)
                yield partiton
            i -= 1
        except StopIteration:
            #print('s', i)
            iterators[i] = None
            i -= 1
            if i < 0:
                return

def get_lambda_source(l):
    src = inspect.getsource(l)
    match = re.search("lambda.*?:(.*)$", src)

    if match is None:
        return l.__name__

    return match.group(1)

# http://stackoverflow.com/questions/12700893/how-to-check-if-a-string-is-a-valid-python-identifier-including-keyword-check
def isidentifier(ident):
    """Determines, if string is valid Python identifier."""

    # Smoke test — if it's not string, then it's not identifier, but we don't
    # want to just silence exception. It's better to fail fast.
    if not isinstance(ident, str):
        raise TypeError('expected str, but got {!r}'.format(type(ident)))

    # Resulting AST of simple identifier is <Module [<Expr <Name "foo">>]>
    try:
        root = ast.parse(ident)
    except SyntaxError:
        return False

    if not isinstance(root, ast.Module):
        return False

    if len(root.body) != 1:
        return False

    if not isinstance(root.body[0], ast.Expr):
        return False

    if not isinstance(root.body[0].value, ast.Name):
        return False

    if root.body[0].value.id != ident:
        return False

    return True

if __name__ == '__main__':
    #print(list(fixed_sum_vector_iter((0,1,1), (8000,2,2), 5)))
    values = ('a', 'a', 'b', 'b', 'c')
    mins = (0, 2)
    maxs = (math.inf, math.inf)
    for p in commutative_partition_iter(values, mins, maxs):
        print(p)