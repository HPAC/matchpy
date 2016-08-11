# -*- coding: utf-8 -*-
import math

from patternmatcher.expressions import Variable, Operation, Arity, Wildcard, Symbol
from patternmatcher.constraints import Constraint, CustomConstraint, EqualVariablesConstraint, MultiConstraint
from patternmatcher.utils import partitions_with_limits, commutative_partition_iter

def linearize(expression, variables=None, constraints=None):
    if variables is None:
        variables = {}
        constraints = {}
        names = set(expression.variables.keys())

        for (name, count) in expression.variables.items():
            variables[name] = [name]

            i = 2
            for _ in range(count - 1):
                newName = name + '_' + str(i)
                while newName in names:
                    i += 1
                    newName = name + '_' + str(i)

                variables[name].append(newName)
                names.add(newName)
                i += 1

            if len(variables[name]) > 1:
                constraints[name] = EqualVariablesConstraint(*variables[name])

    # TODO: Make non-mutating  
    if isinstance(expression, Variable):
        name = expression.name
        expression.name  = variables[name].pop(0)

        if len(variables[name]) == 0 and name in constraints:
            if expression.constraint:
                if not isinstance(expression.constraint, Constraint):
                    expression.constraint = CustomConstraint(expression.constraint)
                
                expression.constraint = MultiConstraint(expression.constraint, constraints[name])
            else:
                expression.constraint = constraints[name]

        linearize(expression.expression, variables, constraints)
    elif isinstance(expression, Operation):
        for operand in expression.operands:
            linearize(operand, variables, constraints)

def match(exprs, pattern, subst=None):
    if subst is None:
        subst = {}
    if isinstance(pattern, Variable):
        if pattern.name in subst:
            if exprs == subst[pattern.name]:
                yield subst
            return
        for newSubst in match(exprs, pattern.expression, subst):
            newSubst = newSubst.copy()
            newSubst[pattern.name] = exprs
            yield newSubst
    elif isinstance(pattern, Wildcard):
        if len(exprs) >= pattern.min_count and len(exprs) <= pattern.max_count:
            yield subst
    elif isinstance(pattern, Symbol):
        if len(exprs) == 1 and exprs[0] == pattern:
            yield subst
    elif isinstance(pattern, Operation):
        if len(exprs) != 1 or not isinstance(exprs[0], Operation):
            return
        yield from _match_operation(exprs[0].operands, pattern, subst)

def _associative_operand_max(operand):
    while isinstance(operand, Variable):
        operand = operand.expression
    if isinstance(operand, Wildcard):
        return math.inf
    return operand.max_count

def _associative_fix_operand_max(parts, maxs, operation):
    newParts = list(parts)
    for i, (part, max_count) in enumerate(zip(parts, maxs)):
        if len(part) > max_count:
            fixed = part[:max_count-1]
            variable = part[max_count-1:]
            newParts[i] = fixed + [operation(*variable)]
    return newParts


def _match_operation(exprs, operation, subst):
    mins = list(o.min_count for o in operation.operands)
    maxs = list(o.max_count for o in operation.operands)
    if operation.associative:
        fake_maxs = list(_associative_operand_max(o) for o in operation.operands)
    else:
        fake_maxs = maxs
    if len(exprs) < sum(mins) or len(exprs) > sum(fake_maxs):
        return
    if len(exprs) == 0:
        yield subst
        return
    limits = list(zip(mins, fake_maxs))
    if operation.commutative:
        parts = commutative_partition_iter(exprs, mins, fake_maxs)
    else:
        parts = partitions_with_limits(exprs, limits)

    for part in parts:
        if operation.associative:
            part = _associative_fix_operand_max(part, maxs, type(operation))
        o_count = len(operation.operands)
        iterators = [None] * o_count
        next_subst = subst
        i = 0
        while True:
            try:
                while i < o_count:
                    if iterators[i] is None:
                        iterators[i] = match(part[i], operation.operands[i], next_subst)
                    next_subst = iterators[i].__next__()
                    i += 1
                yield next_subst
                i -= 1
            except StopIteration:
                iterators[i] = None
                i -= 1
                if i < 0:
                    break

def _main():
    f = Operation.new('f', arity=Arity.binary, associative=True, commutative=True)
    g = Operation.new('g', arity=Arity.unary)
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    x = Variable.dot('x')
    x2 = Variable.dot('x2')
    y = Variable.star('y')
    z = Variable.plus('z')

    expr = f(a, g(b), g(b), g(c), c)
    pattern = f(x, g(Wildcard.dot()), x2)

    for m in match([expr], pattern, {}):
        print('match:')
        for k, v in m.items():
            print('%s: %s' % (k, ', '.join(str(x) for x in v)))

def _main2():
    f = Operation.new('f', Arity.polyadic)
    g = Operation.new('g', Arity.variadic)
    expr = f(Variable.dot('x'), f(Variable.dot('y'), Variable.dot('x')), g(Variable.dot('y')))
    linearize(expr)
    print(expr)

if __name__ == '__main__':
    _main()