# -*- coding: utf-8 -*-
import itertools
import math

from patternmatcher.constraints import (Constraint, CustomConstraint,
                                        EqualVariablesConstraint,
                                        MultiConstraint)
from patternmatcher.expressions import (Arity, Expression, Operation,
                                        Substitution, Symbol, SymbolWildcard,
                                        Variable, Wildcard)
from patternmatcher.utils import (commutative_partition_iter,
                                  partitions_with_limits)

try:
    from backport.typing import (Dict, Iterator, List, Tuple, Union, cast, Sequence, NamedTuple, Callable)
except ImportError:
    from typing import (Dict, Iterator, List, Tuple, Union, cast, Sequence, NamedTuple, Callable)


def linearize(expression, variables=None, constraints=None):
    if variables is None:
        variables = {}
        constraints = {}
        names = set(expression.variables.keys())

        for (name, count) in expression.variables.items():
            variables[name] = [name]

            i = 2
            for _ in range(count - 1):
                new_name = name + '_' + str(i)
                while new_name in names:
                    i += 1
                    new_name = name + '_' + str(i)

                variables[name].append(new_name)
                names.add(new_name)
                i += 1

            if len(variables[name]) > 1:
                constraints[name] = EqualVariablesConstraint(*variables[name])

    # TODO: Make non-mutating
    if isinstance(expression, Variable):
        name = expression.name
        expression.name = variables[name].pop(0)

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
    r"""Tries to match the given `pattern` to the given `expression`.

    Yields each match in form of a substitution that when applied to `pattern` results in the original
    `expression`.

    :param expression: An expression to match.
    :param pattern: The pattern to match.

    :returns: Yields all possible substitutions as dictionaries where each key is the name of the variable
        and the corresponding value is the variables substitution. Applying the substitution to the pattern
        results in the original expression (except for :class:`Wildcard`\s)
    """
    return _match([expression], pattern, {})


def match_anywhere(expression: Expression, pattern: Expression) -> Iterator[Tuple[Substitution, Tuple[int, ...]]]:
    """Tries to match the given `pattern` to the any subexpression of the given `expression`.

    Yields each match in form of a substitution and a position tuple.
    The substitution is a dictionary, where the key is the name of a variable and the value either an expression
    or a list of expressins (iff the variable is a sequence variable).
    When applied to `pattern`, the substitution results in the original matched subexpression.
    The position is a tuple of indices, e.g. the empty tuple refers to the `expression` itself,
    `(0, )` refers to the first child (operand) of the expression, `(0, 0)` to the first child of
    the first child etc.

    :param expression: An expression to match.
    :param pattern: The pattern to match.

    :returns: Yields all possible substitution and position pairs.
    """
    predicate = None
    if pattern.head is not None:
        predicate = lambda x: x.head == pattern.head
    for child, pos in expression.preorder_iter(predicate):
        for subst in _match([child], pattern, {}):
            yield subst, pos

def _match(exprs: List[Expression], pattern: Expression, subst: Substitution) -> Iterator[Substitution]:
    if isinstance(pattern, Variable):
        yield from _match_variable(exprs, pattern, subst, _match)

    elif isinstance(pattern, Wildcard):
        yield from _match_wildcard(exprs, pattern, subst)

    elif isinstance(pattern, Symbol):
        if len(exprs) == 1 and exprs[0] == pattern:
            if pattern.constraint is None or pattern.constraint(subst):
                yield subst

    else:
        assert isinstance(pattern, Operation), 'Unexpected expression of type %r' % type(pattern)
        if len(exprs) != 1 or not isinstance(exprs[0], pattern.__class__):
            return
        op_expr = cast(Operation, exprs[0])
        for result in _match_operation(op_expr.operands, pattern, subst, _match):
            if pattern.constraint is None or pattern.constraint(result):
                yield result

def _match_variable(exprs: List[Expression], variable: Variable, subst: Substitution, matcher: Callable[[List[Expression], Expression, Substitution], Iterator[Substitution]]) -> Iterator[Substitution]:
    inner = variable.expression
    while isinstance(inner, Variable):
        inner = inner.expression
    if len(exprs) == 1 and (not isinstance(inner, Wildcard) or inner.fixed_size):
        expr = exprs[0] # type: Union[Expression,List[Expression]]
    else:
        expr = exprs
    if variable.name in subst:
        if expr == subst[variable.name]:
            if variable.constraint is None or variable.constraint(subst):
                yield subst
        return
    for new_subst in matcher(exprs, variable.expression, subst):
        new_subst = new_subst.copy()
        new_subst[variable.name] = expr
        if variable.constraint is None or variable.constraint(new_subst):
            yield new_subst

def _match_wildcard(exprs: List[Expression], wildcard: Wildcard, subst: Substitution) -> Iterator[Substitution]:
    if wildcard.fixed_size:
        if len(exprs) == wildcard.min_count:
            if isinstance(wildcard, SymbolWildcard) and not isinstance(exprs[0], wildcard.symbol_type):
                return
            if wildcard.constraint is None or wildcard.constraint(subst):
                yield subst
    elif len(exprs) >= wildcard.min_count:
        if wildcard.constraint is None or wildcard.constraint(subst):
            yield subst

def _associative_operand_max(operand):
    while isinstance(operand, Variable):
        operand = operand.expression
    if isinstance(operand, Wildcard):
        return math.inf
    return 1

def _associative_fix_operand_max(parts, maxs, operation):
    new_parts = list(parts)
    for i, (part, max_count) in enumerate(zip(parts, maxs)):
        if len(part) > max_count:
            fixed = part[:max_count-1]
            variable = part[max_count-1:]
            new_parts[i] = fixed + [operation(*variable)]
    return new_parts

def _size(expr):
    while isinstance(expr, Variable):
        expr = expr.expression
    if isinstance(expr, Wildcard):
        return (expr.min_count, (not expr.fixed_size) and math.inf or expr.min_count)
    return (1, 1)

def _match_operation(exprs, operation, subst, matcher):
    if len(operation.operands) == 0:
        if len(exprs) == 0:
            yield subst
        return
    # TODO
    mins, maxs = map(list, zip(*map(_size, operation.operands)))
    if operation.associative:
        fake_maxs = list(_associative_operand_max(o) for o in operation.operands)
    else:
        fake_maxs = maxs
    if len(exprs) < sum(mins) or len(exprs) > sum(fake_maxs):
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
                        iterators[i] = matcher(part[i], operation.operands[i], next_subst)
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
            if isinstance(result, list):
                if len(result) != 1:
                    raise ValueError('Invalid substitution resulted in a variable with multiple expressions.')
                result = result[0]
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


def replace(expression: Expression, position: Sequence[int], replacement: Union[Expression, List[Expression]]) -> Union[Expression, List[Expression]]:
    r"""Replaces the subexpression of `expression` at the given `position` with the given `replacement`.

    The original `expression` itself is not modified, but a modified copy is returned. If the replacement
    is a list of expressions, it will be expanded into the list of operands of the respective operation:

    >>> replace(f(a), (0, ), [b, c])
    f(b, c)

    :param expression: An :class:`Expression` where a (sub)expression is to be replaced.
    :param position: A tuple of indices, e.g. the empty tuple refers to the `expression` itself,
        `(0, )` refers to the first child (operand) of the `expression`, `(0, 0)` to the first
        child of the first child etc.
    :param replacement: Either an :class:`Expression` or a list of :class:`Expression`\s to be
        inserted into the `expression` instead of the original expression at that `position`.
    """
    if position == ():
        return replacement
    if not isinstance(expression, Operation):
        raise IndexError('Invalid position %r for expression %s' % (position, expression))
    if position[0] >= len(expression.operands):
        raise IndexError('Position %r out of range for expression %s' % (position, expression))
    op_class = type(expression)
    pos = position[0]
    subexpr = replace(expression.operands[pos], position[1:], replacement)
    if isinstance(subexpr, list):
        return op_class(*(expression.operands[:pos] + subexpr + expression.operands[pos+1:]))
    operands = expression.operands.copy()
    operands[pos] = subexpr
    return op_class(*operands)

ReplacementRule = NamedTuple('ReplacementRule', [('pattern', Expression), ('replacement', Callable[..., Expression])])

def replace_all(expression: Expression, rules: Sequence[ReplacementRule]) -> Union[Expression, List[Expression]]:
    grouped = itertools.groupby(rules, lambda r: r.pattern.head)
    heads, tmp_groups = map(list, zip(*[(h, list(g)) for h, g in grouped]))
    groups = [list(g) for g in tmp_groups]
    # any_rules = []
    # for i, h in enumerate(heads):
    #     if h is None:
    #         any_rules = groups[i]
    #         del heads[i]
    #         del groups[i]
    #         break
    replaced = True
    while replaced:
        replaced = False
        for head, group in zip(heads, groups):
            predicate = None
            if head is not None:
                predicate = lambda e: e.head == head
            for subexpr, pos in expression.preorder_iter(predicate):
                for pattern, replacement in group:
                    try:
                        subst = next(match(subexpr, pattern))
                        result = replacement(**subst)
                        expression = replace(expression, pos, result)
                        replaced = True
                        break
                    except StopIteration:
                        pass
                if replaced:
                    break
            if replaced:
                break

    return expression

if __name__ == '__main__': # pragma: no cover
    def _main():
        from patternmatcher.utils import match_repr_str
        f = Operation.new('f', arity=Arity.binary)
        g = Operation.new('g', arity=Arity.unary)
        a = Symbol('a')
        b = Symbol('b')
        c = Symbol('c')
        x = Variable.dot('x')

        expr = f(a, g(b), g(g(c), g(a), g(g(b))), g(c), c)
        pattern = g(x)

        for m, pos in match_anywhere(expr, pattern):
            print('match at ', pos, ':')
            print(expr[pos])
            print(match_repr_str(m))

    def _main2():
        f = Operation.new('f', Arity.polyadic)
        g = Operation.new('g', Arity.variadic)
        expr = f(Variable.dot('x'), f(Variable.dot('y'), Variable.dot('x')), g(Variable.dot('y')))
        linearize(expr)
        print(expr)

    def _main3():
        # pylint: disable=invalid-name,bad-continuation
        LAnd = Operation.new('and', Arity.variadic, 'LAnd', associative=True, one_identity=True, commutative=True)
        LOr = Operation.new('or', Arity.variadic, 'LOr', associative=True, one_identity=True, commutative=True)
        LXor = Operation.new('xor', Arity.variadic, 'LXor', associative=True, one_identity=True, commutative=True)
        LNot = Operation.new('not', Arity.unary, 'LNot')
        LImplies = Operation.new('implies', Arity.binary, 'LImplies')
        Iff = Operation.new('iff', Arity.binary, 'Iff')

        x_ = Variable.dot('x')
        x__ = Variable.plus('x')
        y_ = Variable.dot('y')
        y___ = Variable.star('y')
        z_ = Variable.dot('z')
        ___ = Wildcard.star()

        a1 = Symbol('a1')
        a2 = Symbol('a2')
        a3 = Symbol('a3')
        a4 = Symbol('a4')
        a5 = Symbol('a5')
        a6 = Symbol('a6')
        a7 = Symbol('a7')
        a8 = Symbol('a8')
        a9 = Symbol('a9')
        a10 = Symbol('a10')
        a11 = Symbol('a11')

        LBot = Symbol(u'⊥')
        LTop = Symbol(u'⊤')

        expr = LImplies(LAnd(Iff(Iff(LOr(a1, a2), LOr(LNot(a3), Iff(LXor(a4, a5), LNot(LNot(LNot(a6)))))),
        LNot(LAnd(LAnd(a7, a8), LNot(LXor(LXor(LOr(a9, LAnd(a10, a11)), a2), LAnd(LAnd(a11, LXor(a2, Iff(
        a5, a5))), LXor(LXor(a7, a7), Iff(a9, a4)))))))), LImplies(Iff(Iff(LOr(a1, a2), LOr(LNot(a3),
        Iff(LXor(a4, a5), LNot(LNot(LNot(a6)))))), LNot(LAnd(LAnd(a7, a8), LNot(LXor(LXor(LOr(a9, LAnd(
        a10, a11)), a2), LAnd(LAnd(a11, LXor(a2, Iff(a5, a5))), LXor(LXor(a7, a7), Iff(a9, a4)))))))),
        LNot(LAnd(LImplies(LAnd(a1, a2), LNot(LXor(LOr(LOr(LXor(LImplies(LAnd(a3, a4), LImplies(a5, a6)),
        LOr(a7, a8)), LXor(Iff(a9, a10), a11)), LXor(LXor(a2, a2), a7)), Iff(LOr(a4, a9), LXor(LNot(a6),
        a6))))), LNot(Iff(LNot(a11), LNot(a9))))))), LNot(LAnd(LImplies(LAnd(a1, a2), LNot(LXor(LOr(LOr(
        LXor(LImplies(LAnd(a3, a4), LImplies(a5, a6)), LOr(a7, a8)), LXor(Iff(a9, a10), a11)), LXor(
        LXor(a2, a2), a7)), Iff(LOr(a4, a9), LXor(LNot(a6), a6))))), LNot(Iff(LNot(a11), LNot(a9))))))

        rules = [
            # xor(x,⊥) → x
            ReplacementRule(
                LXor(x__, LBot),
                lambda x: LXor(*x)
            ),
            # xor(x, x) → ⊥
            ReplacementRule(
                LXor(x_, x_, ___),
                lambda x: LBot
            ),
            # and(x,⊤) → x
            ReplacementRule(
                LAnd(x__, LTop),
                lambda x: LAnd(*x)
            ),
            # and(x,⊥) → ⊥
            ReplacementRule(
                LAnd(___, LBot),
                lambda: LBot
            ),
            # and(x, x) → x
            ReplacementRule(
                LAnd(x_, x_, y___),
                lambda x, y: LAnd(x, *y)
            ),
            # and(x, xor(y, z)) → xor(and(x, y), and(x, z))
            ReplacementRule(
                LAnd(x_, LXor(y_, z_)),
                lambda x, y, z: LXor(LAnd(x, y), LAnd(x, z))
            ),
            # implies(x, y) → not(xor(x, and(x, y)))
            ReplacementRule(
                LImplies(x_, y_),
                lambda x, y: LNot(LXor(x, LAnd(x, y)))
            ),
            # not(x) → xor(x,⊤)
            ReplacementRule(
                LNot(x_),
                lambda x: LXor(x, LTop)
            ),
            # or(x, y) → xor(and(x, y), xor(x, y))
            ReplacementRule(
                LOr(x_, y_),
                lambda x, y: LXor(LAnd(x, y), LXor(x, y))
            ),
            # iff(x, y) → not(xor(x, y))
            ReplacementRule(
                Iff(x_, y_),
                lambda x, y: LNot(LXor(x, y))
            ),
        ]

        result = replace_all(expr, rules)

        print(result)

    _main3()
