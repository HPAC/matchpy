MatchPy
=======

MatchPy is a pattern matching library for Python.

**Work in progress**

|pypi| |coverage| |build| |docs|

Installation
------------

MatchPy is available via `PyPI <https://pypi.python.org/pypi/matchpy>`_. It can be installed with ``pip install matchpy``.

Overview
--------

This package implements `pattern matching <https://en.wikipedia.org/wiki/Pattern_matching>`_ in Python. Pattern matching is a powerful tool for symbolic computations, operating on symbolic expressions. Given a pattern and an expression (which is usually called *subject*), the goal of pattern matching is to find a substitution for all the variables in the pattern such that the pattern becomes the subject. As an example, consider the pattern :math:`f(x)`, where :math:`f` is a function and :math:`x` is a variable, and the subject :math:`f(a)`, where :math:`a` is a constant symbol. Then the substitution that replaces :math:`x` with :math:`a` is a match. MatchPy supports associative and/or commutative function symbols, as well as sequence variables, similar to pattern matching in `Mathematica <https://reference.wolfram.com/language/guide/Patterns.html>`_. 

A detailed example of how to use MatchPy can be found `here <https://matchpy.readthedocs.io/en/latest/example.html>`_.

MatchPy supports both one-to-one and many-to-one pattern matching. The latter makes use of similarities between patterns to efficiently find matches for multiple patterns at the same time.

The basic algorithms implemented in MachtPy have been described in `this Master thesis <https://arxiv.org/abs/1705.00907>`_.

Expressions
...........

Expressions are tree-like data structures, consisting of operations (functions, internal nodes) and symbols (constants, leaves):

>>> from matchpy import Operation, Symbol, Arity
>>> f = Operation.new('f', Arity.binary)
>>> a = Symbol('a')
>>> print(f(a, a))
f(a, a)

Patterns are expressions which may contain wildcards (variables):

>>> from matchpy import Wildcard
>>> x = Wildcard.dot('x')
>>> print(Pattern(f(a, x)))
f(a, x_)

In the previous example, x is the name of the variable. However, it is also possible to use wildcards without names:

>>> w = Wildcard.dot()
>>> print(Pattern(f(w, w)))
f(_, _)

It is also possible to assign variable names to entire subexpressions:

>>> print(Pattern(f(w, a, variable_name='y')))
y: f(_, a)

Pattern Matching
................

Given a pattern and an expression (which is usually called subject), the idea of pattern matching is to find a substitution that maps wildcards to expressions such that the pattern becomes the subject. In MatchPy, a substitution is a dict that maps variable names to expressions.

>>> from matchpy import match
>>> y = Wildcard.dot('y')
>>> b = Symbol('b')
>>> subject = f(a, b)
>>> pattern = Pattern(f(x, y))
>>> substitution = next(match(subject, pattern))
>>> print(substitution)
{x ↦ a, y ↦ b}

Applying the substitution to the pattern results in the original expression.

>>> from matchpy import substitute
>>> print(substitute(pattern, substitution))
f(a, b)

Sequence Wildcards
..................

Sequence wildcards are wildcards that can match a sequence of expressions instead of just a single expression:

>>> z = Wildcard.plus('z')
>>> pattern = Pattern(f(z))
>>> subject = f(a, b)
>>> substitution = next(match(subject, pattern))
>>> print(substitution)
{z ↦ (a, b)}

Associativity and Commutativity
...............................

MatchPy natively supports associative and/or commutative operations. Nested associative operators are automatically flattened, the operands in commutative operations are sorted:

>>> g = Operation.new('g', Arity.polyadic, associative=True, commutative=True)
>>> print(g(a, g(b, a)))
g(a, a, b)

Associativity and commutativity is also considered for pattern matching:

>>> pattern = Pattern(g(b, x))
>>> subject = g(a, a, b)
>>> print(next(match(subject, pattern)))
{x ↦ g(a, a)}
>>> h = Operation.new('h', Arity.polyadic)
>>> pattern = Pattern(h(b, x))
>>> subject = h(a, a, b)
>>> list(match(subject, pattern))
[]

Many-to-One Matching
....................

There are two classes for many-to-one matching: `DiscriminationNet <https://matchpy.readthedocs.io/en/latest/api/matchpy.matching.syntactic.html>`_ and `ManyToOneMatcher <https://matchpy.readthedocs.io/en/latest/api/matchpy.matching.many_to_one.html>`_. The DiscriminationNet class only supports syntactic pattern matching, that is, operations are neither associative nor commutative. Sequence variables are not supported either. The ManyToOneMatcher class supports associative and/or commutative matching with sequence variables. For syntactic pattern matching, the DiscriminationNet should be used, as it is usually faster.

>>> pattern1 = Pattern(f(a, x))
>>> pattern2 = Pattern(f(y, b))
>>> matcher = ManyToOneMatcher(pattern1, pattern2)
>>> subject = f(a, b)
>>> matches = matcher.match(subject)
>>> for matched_pattern, substitution in sorted(map(lambda m: (str(m[0]), str(m[1])), matches)):
...     print('{} matched with {}'.format(matched_pattern, substitution))
f(a, x_) matched with {x ↦ b}
f(y_, b) matched with {y ↦ a}

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
- Code generation for both one-to-one and many-to-one matching. There is already an experimental implementation, but it still has some dependencies on MatchPy which can probably be removed.
- Improving the documentation with more examples.
- Better test coverage with more randomized tests.
- Implementation of the matching algorithms in a lower-level language, for example C, both for performance and to make MatchPy's functionality available in other languages.

Contributing
------------

If you have some issue or want to contribute, please feel free to open an issue or create a pull request. Help is always appreciated!

The Makefile has several tasks to help development:

- To install all needed packages, you can use ``make init`` .
- To run the tests you can use ``make test``. The tests use `pytest <https://docs.pytest.org/>`_.
- To generate the documentation you can use ``make docs`` .
- To run the style checker (`pylint <https://www.pylint.org/>`_) you can use ``make check`` .

If you have any questions or need help with setting things up, please open an issue and we will try the best to assist you.

.. |pypi| image:: https://img.shields.io/pypi/v/matchpy.svg?style=flat&label=latest%20version
    :target: https://pypi.org/project/matchpy/
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
