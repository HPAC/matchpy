# -*- coding: utf-8 -*-
from typing import TypeVar, Tuple, List, Sequence, Iterator
import itertools
import ast

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

def fixed_sum_vector_iter(minVect : Sequence[int], maxVect : Sequence[int], total : int) -> Iterator[List[int]]:
    assert len(minVect) == len(maxVect), 'len(minVect) != len(maxVect)'
    assert all(minValue <= maxValue for minValue, maxValue in zip(minVect, maxVect)), 'minVect > maxVect'

    minSum = sum(minVect)
    maxSum = sum(maxVect)

    if minSum > total or maxSum < total:
        return

    count = len(maxVect)

    if count <= 1:
        yield len(maxVect) == 1 and [total] or []
        return

    remaining = total - minSum

    realMins = [max(total - maxSum + maximum, minimum) for minimum, maximum in zip(minVect, maxVect)]
    realMaxs = [min(remaining + minimum, maximum) for minimum, maximum in zip(minVect, maxVect)]

    values = list(realMins)

    remaining = total - sum(realMins)

    j = count - 1
    while remaining > 0:
        toAdd = min(realMaxs[j] - realMins[j], remaining)
        values[j] += toAdd
        remaining -= toAdd
        j -= 1

    while True:
        pos = len(realMins) - 2
        yield values[:]
        while True:
            values[pos] += 1
            values[-1] -= 1
            if values[-1] < realMins[-1] or values[pos] > realMaxs[pos]:
                if pos == 0:
                    return
                values[pos] = realMins[pos] # reset current position
                values[-1] = total - sum(values[:-1]) # reset last position
                pos -= 1
            else:
                break

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
    print(list(fixed_sum_vector_iter((0,1,1), (8000,2,2), 5)))