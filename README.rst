matchpy
==============

A pattern matching libary for python.

**Work in progress**

|coverage| |build| |docs|

Overview
--------

This package implements `pattern matching <https://en.wikipedia.org/wiki/Pattern_matching>`_ in python. It is similar
to the implementation in `Mathematica <https://reference.wolfram.com/language/guide/Patterns.html>`_ or
`Haskell <https://www.haskell.org/tutorial/patterns.html>`_.

Expressions
...........

Expressions and patterns both have a tree structure. Expressions consist of symbols (leafs) and operations
(internal nodes)::

    >>> from matchpy.expressions import Operation, Symbol, Arity
    >>> f = Operation.new('f', Arity.binary)
    >>> a = Symbol('a')
    >>> print(f(a, a))
    f(a, a)

Patterns are expressions which can additionally contain variables and wildcards. Variables can give a
name to a node of the pattern so that it can be accessed later. Wildcards are placeholders that stand for any
expression. Usually, the two are used in combination::

    >>> from matchpy.expressions import Variable
    >>> x = Variable.dot('x')
    >>> print(f(a, x))
    f(a, x_)

However, unnamed wildcards can also be used::

    >>> from matchpy.expressions import Wildcard
    >>> w = Wildcard.dot()
    >>> print(f(w, w))
    f(_, _)

Or a more complex expression can be named with a variable::

    >>> print(Variable('y', f(w, a)))
    y: f(_, a)

In addition, it supports sequence wildcards that stand for multiple expressions::

    >>> z = Variable.plus('z')
    >>> print(f(z))
    f(z__)


Substitutions
.............

Matches are given in the form of substitutions, which are a mapping from variable names to expressions::

    >>> from matchpy.matching.one_to_one import match
    >>> y = Variable.dot('y')
    >>> b = Symbol('b')
    >>> expression = f(a, b)
    >>> pattern = f(x, y)
    >>> substitution = next(match(expression, pattern))
    >>> substitution
    {'x': FrozenSymbol('a'), 'y': FrozenSymbol('b')}

Replacing the variables in the pattern according to the substitution will yield the original subject expression::

    >>> from matchpy.functions import substitute
    >>> original, _ = substitute(pattern, substitution)
    >>> print(original)
    f(a, b)


.. |coverage| image:: https://coveralls.io/repos/github/HPAC/matchpy/badge.svg?branch=master
    :target: https://coveralls.io/github/HPAC/matchpy?branch=master
    :alt: Test coverage

.. |build| image:: https://travis-ci.org/HPAC/matchpy.svg?branch=master
    :target: https://travis-ci.org/HPAC/matchpy
    :alt: Build status of the master branch

.. |docs| image:: https://readthedocs.org/projects/matchpy/badge/?version=latest
    :target: http://matchpy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
