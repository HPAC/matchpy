matchpy
=======

A pattern matching libary for python.

**Work in progress**

|pypi| |coverage| |build| |docs|

Overview
--------

This package implements `pattern matching <https://en.wikipedia.org/wiki/Pattern_matching>`_ in python. It is similar
to the implementation in `Mathematica <https://reference.wolfram.com/language/guide/Patterns.html>`_.
A `detailed example <https://matchpy.readthedocs.io/en/latest/example.html>`_ of how you can use matchpy can be found
in the `documentation <https://matchpy.readthedocs.io/en/latest/>`_.

In addition to the basic matching algorithm, there are data structures that can be used for more efficient many-to-one
matching like the `ManyToOneMatcher <https://matchpy.readthedocs.io/en/latest/api/matchpy.matching.many_to_one.html>`_
and the `DiscriminationNet <https://matchpy.readthedocs.io/en/latest/api/matchpy.matching.syntactic.html>`_.

Expressions
...........

Expressions and patterns both have a tree structure. Expressions consist of symbols (leafs) and operations
(internal nodes):

>>> from matchpy import Operation, Symbol, Arity
>>> f = Operation.new('f', Arity.binary)
>>> a = Symbol('a')
>>> print(f(a, a))
f(a, a)

Patterns are expressions which can additionally contain wildcards and subexpressions can have a variable name assigned
to them. During matching, a subject matching a pattern with a variable will be captured so it can be accessed later.
Wildcards are placeholders that stand for any expression. Usually, the wildcards are used in combination with a variable
name:

>>> from matchpy import Wildcard
>>> x = Wildcard.dot('x')
>>> print(Pattern(f(a, x)))
f(a, x_)

Here x is the name of the variable. However, unnamed wildcards can also be used:

>>> w = Wildcard.dot()
>>> print(Pattern(f(w, w)))
f(_, _)

Or a more complex expression can be named with a variable:

>>> print(Pattern(f(w, a, variable_name='y')))
y: f(_, a)

In addition, sequence wildcards that can match for multiple expressions are supported:

>>> z = Wildcard.plus('z')
>>> print(Pattern(f(z)))
f(z__)


Substitutions
.............

Matches are given in the form of substitutions, which are a mapping from variable names to expressions:

>>> from matchpy import match
>>> y = Wildcard.dot('y')
>>> b = Symbol('b')
>>> expression = f(a, b)
>>> pattern = Pattern(f(x, y))
>>> substitution = next(match(expression, pattern))
>>> substitution
{'x': Symbol('a'), 'y': Symbol('b')}

Replacing the variables in the pattern according to the substitution will yield the original subject expression:

>>> from matchpy import substitute
>>> print(substitute(pattern, substitution))
f(a, b)


.. |pypi| image:: https://img.shields.io/pypi/v/matchpy.svg?style=flat-square&label=latest%20version
    :target: https://pypi.python.org/pypi/matchpy
    :alt: Latest version released on PyPi

.. |coverage| image:: https://coveralls.io/repos/github/HPAC/matchpy/badge.svg?branch=master
    :target: https://coveralls.io/github/HPAC/matchpy?branch=master
    :alt: Test coverage

.. |build| image:: https://travis-ci.org/HPAC/matchpy.svg?branch=master
    :target: https://travis-ci.org/HPAC/matchpy
    :alt: Build status of the master branch

.. |docs| image:: https://readthedocs.org/projects/matchpy/badge/?version=latest
    :target: https://matchpy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
