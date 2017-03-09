# -*- coding: utf-8 -*-
import pytest

from matchpy.expressions.expressions import Operation, Symbol, Arity, Wildcard, make_dot_variable, make_star_variable, make_plus_variable
import matchpy

@pytest.fixture(autouse=True)
def add_default_expressions(doctest_namespace):
    doctest_namespace['f'] = Operation.new('f', Arity.variadic)
    doctest_namespace['a'] = Symbol('a')
    doctest_namespace['b'] = Symbol('b')
    doctest_namespace['c'] = Symbol('c')
    doctest_namespace['x_'] = make_dot_variable('x')
    doctest_namespace['y_'] = make_dot_variable('y')
    doctest_namespace['_'] = Wildcard.dot()
    doctest_namespace['__'] = Wildcard.plus()
    doctest_namespace['___'] = Wildcard.star()
    doctest_namespace['__name__'] = '__main__'

    for name in matchpy.__all__:
        doctest_namespace[name] = getattr(matchpy, name)
