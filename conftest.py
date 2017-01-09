# -*- coding: utf-8 -*-
import pytest

from matchpy.expressions import Operation, Symbol, Variable, Arity, Wildcard
import matchpy

@pytest.fixture(autouse=True)
def add_default_expressions(doctest_namespace):
    doctest_namespace['f'] = Operation.new('f', Arity.variadic)
    doctest_namespace['a'] = Symbol('a')
    doctest_namespace['b'] = Symbol('b')
    doctest_namespace['c'] = Symbol('c')
    doctest_namespace['x_'] = Variable.dot('x')
    doctest_namespace['y_'] = Variable.dot('y')
    doctest_namespace['_'] = Wildcard.dot()
    doctest_namespace['__'] = Wildcard.plus()
    doctest_namespace['___'] = Wildcard.star()
    doctest_namespace['__name__'] = '__main__'

    for name in matchpy.__all__:
        doctest_namespace[name] = getattr(matchpy, name)
