# -*- coding: utf-8 -*-

from typing import cast, Dict, Set

from patternmatcher.expressions import (Arity, Expression, Operation, Symbol,
                                        Variable, Wildcard)
from patternmatcher.functions import ReplacementRule
from patternmatcher.syntactic import DiscriminationNet


class ManyToOneMatcher(object):
    def __init__(self, *patterns: Expression) -> None:
        self.patterns = patterns
        self.graphs = {} # type: Dict[type,Set[Expression]]
        self.nets = {} # type: Dict[type,DiscriminationNet]

        for pattern in patterns:
            for expression, _ in pattern.preorder_iter(lambda e: isinstance(e, Operation) and e.commutative):
                operation = cast(Operation, expression)
                if type(operation) not in self.graphs:
                    self.graphs[type(operation)] = set()
                expressions = [o for o in operation.operands if o.is_syntactic and o.head is not None]
                self.graphs[type(operation)].update(expressions)

        for g, exprs in self.graphs.items():
            net = DiscriminationNet()
            for expr in exprs:
                net.add(expr)
            self.nets[g] = net


if __name__ == '__main__': # pragma: no cover
    def _main():
        # pylint: disable=invalid-name,bad-continuation
        Times = Operation.new('*', Arity.variadic, 'Times', associative=True, one_identity=True)
        Plus = Operation.new('+', Arity.variadic, 'Plus', associative=True, one_identity=True, commutative=True)
        Minus = Operation.new('-', Arity.unary, 'Minus')
        Inv = Operation.new('Inv', Arity.unary, 'Inv')
        Trans = Operation.new('TLeft', Arity.unary, 'Trans')

        x_ = Variable.dot('x')
        y_ = Variable.dot('y')
        y___ = Variable.star('y')
        z___ = Variable.star('z')
        ___ = Wildcard.star()

        Zero = Symbol('0')
        Identity = Symbol('I')

        rules = [
            # --x -> x
            ReplacementRule(
                Minus(Minus(x_)),
                lambda x: x
            ),
            # -0 -> 0
            ReplacementRule(
                Minus(Zero),
                lambda: Zero
            ),
            # x + 0 -> x
            ReplacementRule(
                Plus(x_, Zero),
                lambda x: x
            ),
            # y + x - x -> y
            ReplacementRule(
                Plus(y_, x_, Minus(x_)),
                lambda x, y: y
            ),
            # x * 0 -> 0
            ReplacementRule(
                Times(___, Zero, ___),
                lambda: Zero
            ),
            # x * x^-1 -> I
            ReplacementRule(
                Times(y___, x_, Inv(x_), z___),
                lambda x, y, z: Times(*(y + [Identity] + z))
            ),
            # TLeft(x) * TLeft(x^-1) -> I
            ReplacementRule(
                Times(y___, Trans(x_), Trans(Inv(x_)), z___),
                lambda x, y, z: Times(*(y + [Identity] + z))
            ),
            # TLeft(TLeft(x)) -> x
            ReplacementRule(
                Trans(Trans(x_)),
                lambda x: x
            ),
            # TLeft(0) -> 0
            ReplacementRule(
                Trans(Zero),
                lambda: Zero
            ),
            # TLeft(I) -> I
            ReplacementRule(
                Trans(Identity),
                lambda: Identity
            ),
            # ((x)^-1)^-1 -> x
            ReplacementRule(
                Inv(Inv(x_)),
                lambda x: x
            ),
            # x * I -> x
            ReplacementRule(
                Times(y___, x_, Identity, z___),
                lambda x, y, z: Times(*(y + [x] + z))
            ),
            # I * x -> x
            ReplacementRule(
                Times(y___, Identity, x_, z___),
                lambda x, y, z: Times(*(y + [x] + z))
            ),
        ]

        ManyToOneMatcher(*(r.pattern for r in rules))

    _main()
