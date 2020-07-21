MatchPy
=======

MatchPy is a library for pattern matching on symbolic expressions in Python.

**Work in progress**

|pypi| |conda| |coverage| |build| |docs| |joss| |doi|

Installation
------------

MatchPy is available via `PyPI <https://pypi.python.org/pypi/matchpy>`_, and for Conda via `conda-forge <https://anaconda.org/conda-forge/matchpy>`_. It can be installed with ``pip install matchpy`` or ``conda install -c conda-forge matchpy``.

Overview
--------

This package implements `pattern matching <https://en.wikipedia.org/wiki/Pattern_matching>`_ in Python. Pattern matching is a powerful tool for symbolic computations, operating on symbolic expressions. Given a pattern and an expression (which is usually called *subject*), the goal of pattern matching is to find a substitution for all the variables in the pattern such that the pattern becomes the subject. As an example, consider the pattern ``f(x)``, where ``f`` is a function and ``x`` is a variable, and the subject ``f(a)``, where ``a`` is a constant symbol. Then the substitution that replaces ``x`` with ``a`` is a match. MatchPy supports associative and/or commutative function symbols, as well as sequence variables, similar to pattern matching in `Mathematica <https://reference.wolfram.com/language/guide/Patterns.html>`_. 

A detailed example of how to use MatchPy can be found `here <https://matchpy.readthedocs.io/en/latest/example.html>`_.

MatchPy supports both one-to-one and many-to-one pattern matching. The latter makes use of similarities between patterns to efficiently find matches for multiple patterns at the same time.

A list of publications about MatchPy can be found `below <Publications_>`_.

Expressions
...........

Expressions are tree-like data structures, consisting of operations (functions, internal nodes) and symbols (constants, leaves):

>>> from matchpy import Operation, Symbol, Arity
>>> f = Operation.new('f', Arity.binary)
>>> a = Symbol('a')
>>> print(f(a, a))
f(a, a)

Patterns are expressions which may contain wildcards (variables):

>>> from matchpy import Pattern, Wildcard
>>> x = Wildcard.dot('x')
>>> print(Pattern(f(a, x)))
f(a, x_)

In the previous example, ``x`` is the name of the variable. However, it is also possible to use wildcards without names:

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

When a fixed set of patterns is matched repeatedly against different subjects, matching can be sped up significantly by using many-to-one matching. The idea of many-to-one matching is to construct a so called discrimination net, a data structure similar to a decision tree or a finite automaton that exploits similarities between patterns. In MatchPy, there are two such data structures, implemented as classes: `DiscriminationNet <https://matchpy.readthedocs.io/en/latest/api/matchpy.matching.syntactic.html>`_ and `ManyToOneMatcher <https://matchpy.readthedocs.io/en/latest/api/matchpy.matching.many_to_one.html>`_. The DiscriminationNet class only supports syntactic pattern matching, that is, operations are neither associative nor commutative. Sequence variables are not supported either. The ManyToOneMatcher class supports associative and/or commutative matching with sequence variables. For syntactic pattern matching, the DiscriminationNet should be used, as it is usually faster.

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

Publications
------------

| `MatchPy: Pattern Matching in Python <http://joss.theoj.org/papers/10.21105/joss.00670>`_
| Manuel Krebber and Henrik Barthels
| Journal of Open Source Software, Volume 3(26), pp. 2, June 2018.
|

| `Efficient Pattern Matching in Python <https://dl.acm.org/citation.cfm?id=3149871>`_
| Manuel Krebber, Henrik Barthels and Paolo Bientinesi
| Proceedings of the 7th Workshop on Python for High-Performance and Scientific Computing, November 2017.
|

| `MatchPy: A Pattern Matching Library <http://conference.scipy.org/proceedings/scipy2017/manuel_krebber.html>`_
| Manuel Krebber, Henrik Barthels and Paolo Bientinesi
| Proceedings of the 15th Python in Science Conference, July 2017.
|

| `Non-linear Associative-Commutative Many-to-One Pattern Matching with Sequence Variables <https://arxiv.org/abs/1705.00907>`_
| Manuel Krebber
| Master Thesis, RWTH Aachen University, May 2017
|

If you want to cite MatchPy, please reference the JOSS paper::

    @article{krebber2018,
        author    = {Manuel Krebber and Henrik Barthels},
        title     = {{M}atch{P}y: {P}attern {M}atching in {P}ython},
        journal   = {Journal of Open Source Software},
        year      = 2018,
        pages     = 2,
        month     = jun,
        volume    = {3},
        number    = {26},
        doi       = "10.21105/joss.00670",
        web       = "http://joss.theoj.org/papers/10.21105/joss.00670",
    }

.. |pypi| image:: https://img.shields.io/pypi/v/matchpy.svg?style=flat
    :target: https://pypi.org/project/matchpy/
    :alt: Latest version released on PyPi

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/matchpy.svg
    :target: https://anaconda.org/conda-forge/matchpy
    :alt: Latest version released via conda-forge

.. |coverage| image:: https://coveralls.io/repos/github/HPAC/matchpy/badge.svg?branch=master
    :target: https://coveralls.io/github/HPAC/matchpy?branch=master
    :alt: Test coverage

.. |build| image:: https://travis-ci.org/HPAC/matchpy.svg?branch=master
    :target: https://travis-ci.org/HPAC/matchpy
    :alt: Build status of the master branch

.. |docs| image:: https://readthedocs.org/projects/matchpy/badge/?version=latest
    :target: https://matchpy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
    
.. |joss| image:: http://joss.theoj.org/papers/e456bc05880b533652980aee6550a3cb/status.svg
    :target: http://joss.theoj.org/papers/e456bc05880b533652980aee6550a3cb
    :alt: The Journal of Open Source Software
    
.. |doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1294930.svg
   :target: https://doi.org/10.5281/zenodo.1294930
   :alt: Digital Object Identifier
