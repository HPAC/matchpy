MatchPy
=======

MatchPy is a pattern matching libary for python.

**Work in progress**

|pypi| |coverage| |build| |docs|

Installation
------------

MatchPy is availiablle via `PyPI <https://pypi.python.org/pypi/matchpy>`_. You can install it using ``pip install matchpy``.

Overview
--------

This package implements `pattern matching <https://en.wikipedia.org/wiki/Pattern_matching>`_ in python. It is similar
to the implementation in `Mathematica <https://reference.wolfram.com/language/guide/Patterns.html>`_.
A `detailed example <https://matchpy.readthedocs.io/en/latest/example.html>`_ of how you can use matchpy can be found
in the `documentation <https://matchpy.readthedocs.io/en/latest/>`_.
Some of the implemented algorithms have been described in `this Master thesis <https://arxiv.org/abs/1705.00907>`_.

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


Roadmap
-------

Besides the existing features, we plan on adding the following to MatchPy:

- Support for Mathematica's ``Alternatives``: For example ``f(a | b)`` would match either ``f(a)`` or ``f(b)``.
- Support for Mathematica's ``Repeated``: For example ``f(a..)`` would match ``f(a)``, ``f(a, a)``, ``f(a, a, a)``, etc.
- Support pattern sequences (``PatternSequence`` in Mathematica). These are mainly useful in combination with
  ``Alternatives`` or ``Repeated``, e.g. ``f(a | (b, c))`` would match either ``f(a)`` or ``f(b, c)``.
  ``f((a a)..)`` would match any ``f`` with an even number of ``a`` arguments.
- All these additional pattern features need to be supported in the ``ManyToOneMatcher`` as well.
- Better integration with existing types such as ``dict``.
- Code generation for both one-to-one and many-to-one matching.
- Improving the documentation with more examples.
- Better test coverage with more randomized tests.

Contributing
------------

If you have some issue or want to contribute, please feel free to open an issue or create a pull request. Help is always appreciated!

The Makefile has several tasks to help development:

- To install all needed packages, you can use ``make init`` .
- To run the tests you can use ``make test``. The tests use `pytest <https://docs.pytest.org/>`_.
- To generate the documentation you can use ``make docs`` .
- To run the style checker (`pylint <https://www.pylint.org/>`_) you can use ``make check`` .

If you have any questions or need help with setting things up, please open an issue and we will try the best to assist you.

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
