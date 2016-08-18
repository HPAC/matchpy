# -*- coding: utf-8 -*-
import math
from typing import Dict, Tuple, Union, List, Iterator

from patternmatcher.expressions import Variable, Operation, Arity, Wildcard, Symbol, Expression
from patternmatcher.constraints import Constraint, CustomConstraint, EqualVariablesConstraint, MultiConstraint
from patternmatcher.utils import partitions_with_limits, commutative_partition_iter

Substitution = Dict[str, Union[Expression, List[Expression]]]

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


def match(expression: Expression, pattern: Expression) -> Iterator[Substitution]:
    """Tries to match the given `pattern` to the given `expression`.

    Yields each match in form of a substitution that when applied to `pattern` results in the original
    `expression`. 

    :param expression: An expression to match.
    :param pattern: The pattern to match.

    :returns: Yields all possible substitutions as dictionaries where each key is the name of the variable
        and the corresponding value is the variables substitution. Applying the substitution to the pattern
        results in the original expression (except for :class:`Wildcard`s)
    """
    return _match([expression], pattern, {})

def _match(exprs: List[Expression], pattern: Expression, subst: Substitution) -> Iterator[Substitution]:
    if isinstance(pattern, Variable):
        wc = pattern.expression
        while isinstance(wc, Variable):
            wc = wc.expression
        if isinstance(wc, Wildcard) and wc.min_count == 1 and wc.fixed_size:
            expr = exprs[0]
        else:
            expr = exprs
        if pattern.name in subst:
            if expr == subst[pattern.name]:
                if pattern.constraint is None or pattern.constraint(subst):
                    yield subst
            return
        for newSubst in _match(exprs, pattern.expression, subst):
            newSubst = newSubst.copy()
            newSubst[pattern.name] = expr
            if pattern.constraint is None or pattern.constraint(newSubst):
                yield newSubst

    elif isinstance(pattern, Wildcard):
        if pattern.fixed_size:
            if len(exprs) == pattern.min_count:
                if pattern.constraint is None or pattern.constraint(subst):
                    yield subst
        elif len(exprs) >= pattern.min_count:
            if pattern.constraint is None or pattern.constraint(subst):
                yield subst

    elif isinstance(pattern, Symbol):
        if len(exprs) == 1 and exprs[0] == pattern:
            if pattern.constraint is None or pattern.constraint(subst):
                yield subst

    elif isinstance(pattern, Operation):
        if len(exprs) != 1 or type(exprs[0]) != type(pattern):
            return
        for result in _match_operation(exprs[0].operands, pattern, subst):            
            if pattern.constraint is None or pattern.constraint(result):
                yield result 

def _associative_operand_max(operand):
    while isinstance(operand, Variable):
        operand = operand.expression
    if isinstance(operand, Wildcard):
        return math.inf
    return 1

def _associative_fix_operand_max(parts, maxs, operation):
    newParts = list(parts)
    for i, (part, max_count) in enumerate(zip(parts, maxs)):
        if len(part) > max_count:
            fixed = part[:max_count-1]
            variable = part[max_count-1:]
            newParts[i] = fixed + [operation(*variable)]
    return newParts

def _size(expr):
    while isinstance(expr, Variable):
        expr = expr.expression
    if isinstance(expr, Wildcard):
        return (expr.min_count, (not expr.fixed_size) and math.inf or expr.min_count)
    return (1, 1)

def _match_operation(exprs, operation, subst):
    # TODO
    mins, maxs = map(list, zip(*map(_size, operation.operands)))
    if operation.associative:
        fake_maxs = list(_associative_operand_max(o) for o in operation.operands)
    else:
        fake_maxs = maxs
    if len(exprs) < sum(mins) or len(exprs) > sum(fake_maxs):
        return
    if len(operation.operands) == 0:
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
                        iterators[i] = _match(part[i], operation.operands[i], next_subst)
                    next_subst = iterators[i].__next__()
                    i += 1
                yield next_subst
                i -= 1
            except StopIteration:
                iterators[i] = None
                i -= 1
                if i < 0:
                    break

def substitute(expression: Expression, substitution: Substitution) -> Tuple[Union[Expression, List[Expression]], bool]:
    """Replaces variables in the given `expression` by the given `substitution`.
    
    In addition to the resulting expression(s), a bool is returned indicating whether anything was substituted.
    If nothing was substituted, the original expression is returned.
    Not that this function returns a list of expressions iff the expression is a variable and its substitution
    is a list of expressions. In other cases were a substitution is a list of expressions, the expressions will
    be integrated as operands in the surrounding operation:

    >>> substitute(f(x, c), {'x': [a, b]}})
    f(a, b, c), True

    :param expression: An expression in which variables are substituted.
    :param substitution: A substitution dictionary. The key is the name of the variable,
        the value either an expression or a list of expression to use as a replacement for
        the variable.
    """
    if isinstance(expression, Variable):
        if expression.name in substitution:
            return substitution[expression.name], True
        result, replaced = substitute(expression.expression, substitution)
        if replaced:
            return Variable(expression.name, result), True
    elif isinstance(expression, Operation):
        any_replaced = False
        new_operands = []
        for operand in expression.operands:
            result, replaced = substitute(operand, substitution)
            if replaced:
                any_replaced = True 
            if isinstance(result, list):
                new_operands.extend(result)
            else:
                new_operands.append(result)
        if any_replaced:
            return type(expression)(*new_operands), True

    return expression, False

def _main():
    from patternmatcher.utils import match_repr_str
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

    for m in match(expr, pattern):
        print('match:')
        print(match_repr_str(m))

def _main2():
    f = Operation.new('f', Arity.polyadic)
    g = Operation.new('g', Arity.variadic)
    expr = f(Variable.dot('x'), f(Variable.dot('y'), Variable.dot('x')), g(Variable.dot('y')))
    linearize(expr)
    print(expr)

if __name__ == '__main__':
    _main()