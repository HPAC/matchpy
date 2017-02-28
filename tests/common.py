# -*- coding: utf-8 -*-
from matchpy.expressions.expressions import (Arity, Operation, Symbol, Wildcard, SymbolWildcard)

from .utils import MockConstraint


class SpecialSymbol(Symbol):
    pass


f = Operation.new('f', Arity.variadic)
f2 = Operation.new('f2', Arity.variadic)
f_u = Operation.new('f_u', Arity.unary)
f_i = Operation.new('f_i', Arity.variadic, one_identity=True)
f_c = Operation.new('f_c', Arity.variadic, commutative=True)
f2_c = Operation.new('f2_c', Arity.variadic, commutative=True)
f_a = Operation.new('f_a', Arity.variadic, associative=True)
f_ac = Operation.new('f_ac', Arity.variadic, associative=True, commutative=True)
a = Symbol('a')
b = Symbol('b')
c = Symbol('c')
d = Symbol('d')
s = SpecialSymbol('s')
_ = Wildcard.dot()
_s = Wildcard.symbol()
_ss = Wildcard.symbol(SpecialSymbol)
x_ = Wildcard.dot('x')
s_ = Wildcard.symbol('s')
ss_ = Wildcard.symbol('ss', SpecialSymbol)
y_ = Wildcard.dot('y')
z_ = Wildcard.dot('z')
__ = Wildcard.plus()
x__ = Wildcard.plus('x')
y__ = Wildcard.plus('y')
z__ = Wildcard.plus('z')
___ = Wildcard.star()
x___ = Wildcard.star('x')
y___ = Wildcard.star('y')
z___ = Wildcard.star('z')

mock_constraint_false = MockConstraint(False)
mock_constraint_true = MockConstraint(True)

del Arity
del Operation
del Symbol
del Wildcard
del MockConstraint

__all__ = [name for name in dir() if not name.startswith('__') or all(c == '_' for c in name)]
