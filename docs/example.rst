**********************
Linear Algebra Example
**********************

As an example, we will write the classes necessary to construct linear algebra equations.
These equations consist of scalars, vectors, and matrices, as well as multiplication, addition,
transposition, and inversion.

Lets start by importing everything we need:

>>> from matchpy import *

Symbols
-------

First off, we create simple classes for our scalars and vectors:

>>> class Scalar(Symbol):
...     pass
>>> class Vector(Symbol):
...     pass

Now we can create vectors and scalars like this:

>>> a = Scalar('a')
>>> v = Vector('v')

For matrices, we want to be able to specify additional properties that a matrix has, for
example it might be a diagonal or triangular matrix. We will just use a set of strings for the properties:

>>> class Matrix(Symbol):
...     def __init__(self, name, properties=[]):
...         super().__init__(name)
...         self.properties = frozenset(properties)

Now we can create matrices like this:

>>> M1 = Matrix('M1', ['diagonal', 'square'])
>>> M2 = Matrix('M2', ['symmetric', 'square'])
>>> M3 = Matrix('M3', ['triangular'])

Operations
----------

We can quickly create a new operation using the `.Operation.new` factory method:

>>> Times = Operation.new('*', Arity.variadic, 'Times', associative=True, one_identity=True, infix=True)

We need to specify a name (``'*'``) and arity for the operation. In case that the name is not a valid python identifier,
we also need to specify a class name (``'Times'``). The matrix multiplication is associative, but not commutative.
In addition, we set *one_identity* to ``True``, which means that a multiplication with a single operand can be replaced
by that operand:

>>> Times(a)
Scalar('a')

The infix property is used when printing terms so that they look prettier:

>>> print(Times(a, v))
(a * v)

An alternative way of adding a new operation, is creating a subclass of `.Operation` manually.
This is especially useful, if you want to add custom methods or properties to your operations.
For example, we can customize the string formatting of the transposition:

>>> class Transpose(Operation):
...     name = '^T'
...     arity = Arity.unary
...     def __str__(self):
...         return '({})^T'.format(self.operands[0])

Lets define the remaining operations:

>>> Plus = Operation.new('+', Arity.variadic, 'Plus', one_identity=True, infix=True, commutative=True, associative=True)
>>> Inverse = Operation.new('I', Arity.unary, 'Inverse')

Finally, we can compose more complex terms:

>>> print(Plus(Times(v, Transpose(v)), Times(a, Inverse(M1))))
((a * I(M1)) + (v * (v)^T))

Note that the summands are automatically sorted, because *Plus* is commutative.

Wildcards and Variables
-----------------------

In patterns, we can use `wildcards <.Wildcard>` as a placehold that match anything:

>>> _ = Wildcard.dot()
>>> is_match(a, Pattern(_))
True

However, for our linear algebra patterns, we want to distinguish between different kinds of symbols.
Hence, we can make use of `symbol wildcards <.SymbolWildcard>`, e.g. to create a wildcard that only matches vectors:

>>> _v = Wildcard.symbol(Vector)
>>> is_match(a, Pattern(_v))
False
>>> is_match(v, Pattern(_v))
True

We can also assign a name to wildcards and in that case, we call them variables. These names are used to
populate the match substitution in case there is a match:

>>> x_ = Wildcard.dot('x')
>>> next(match(Times(a, v), Pattern(Times(x_, _v))))
{'x': Scalar('a')}

Constraints
-----------

Patterns can be limited in what is matched by adding constraints. A constraint is essentially a callback,
that gets the match substitution and can return either ``True`` or ``False``. You can either use the `.CustomConstraint`
class with any (lambda) function, or create your own subclass of `.Constraint`.

For example, if we want to only match diagonal matrices with a certain variable, we can create a constraint for that:

>>> C_ = Wildcard.symbol('M3', Matrix)
>>> C_is_diagonal_matrix = CustomConstraint(lambda M3: 'diagonal' in M3.properties)
>>> pattern = Pattern(C_, C_is_diagonal_matrix)

Then the variable *M3* will only match diagonal matrices:

>>> is_match(M1, pattern)
True
>>> is_match(M2, pattern)
False

Example: Simplifying multiplication with inverse matrix
-------------------------------------------------------

Now, we can build patterns to find whatever subexpressions we are interested in. For example, we could remove all
occurences of a matrix being multiplied with its inverse. For that we need sequence wildcards. Instead of
matching a single term, they can match a sequence of terms. We can create sequence variables like this:

>>> ctx1 = Wildcard.plus('ctx1')
>>> ctx2 = Wildcard.star('ctx2')

``ctx1`` is a plus variables and matches a sequence one or more terms. ``ctx2`` is a star variables and can match any
sequence of terms, including the empty one. With these sequence variables, we can create the rules:

>>> x = Wildcard.dot('x')
>>> simplify_matrix_inverse_rules = [
...     ReplacementRule(
...         Pattern(Times(ctx1, x, Inverse(x), ctx2)),
...         lambda ctx1, ctx2, x: Times(*ctx1, *ctx2)
...     ),
...     ReplacementRule(
...         Pattern(Times(ctx2, x, Inverse(x), ctx1)),
...         lambda ctx1, ctx2, x: Times(*ctx2, *ctx1)
...     )
... ]

We need two variations of the rule to make sure that we do not accidentially create an empty product. In the first
rule, at least one operand must preceed the inverse pair. In the second one, at least one operand must come after it.

For the actual replacement, we can use the `.replace_all` function:

>>> expr = Times(M1, Inverse(M1), M2)
>>> replace_all(expr, simplify_matrix_inverse_rules)
Matrix('M2')

For the case that there are no other factors in the product, we can add another rule that replaces
it with the identity matrix:

>>> Identity = Matrix('I')
>>> simplify_matrix_inverse_rules.append(
...     ReplacementRule(
...         Pattern(Times(x, Inverse(x))),
...         lambda x: Identity
...     )
... )

Lets see this new rule in action:

>>> expr2 = Times(M1, Inverse(M1))
>>> replace_all(expr2, simplify_matrix_inverse_rules)
Matrix('I')

Because ``Times`` is associative, these rules even work for more complex expressions:

>>> expr3 = Times(M1, M1, M2, Inverse(Times(M1, M2)), M2)
>>> replace_all(expr3, simplify_matrix_inverse_rules)
Times(Matrix('M1'), Matrix('M2'))

Note that we can normalize a matrix product inside an inversion by moving it outside, i.e.
using the equality :math:`(A B)^{-1} = B^{-1} A^{-1}`:

>>> y = Wildcard.dot('y')
>>> simplify_matrix_inverse_rules.append(
...     ReplacementRule(
...         Pattern(Inverse(Times(x, y))),
...         lambda x, y: Times(Inverse(y), Inverse(x))
...     )
... )

This allows us to simplify an expression like this:

>>> expr4 = Times(M1, M2, Inverse(Times(M3, M1, M2)))
>>> replace_all(expr4, simplify_matrix_inverse_rules)
Inverse(Matrix('M3'))

Or this:

>>> expr5 = Times(M1, M2, Inverse(Times(M3, M2)))
>>> replace_all(expr5, simplify_matrix_inverse_rules)
Times(Matrix('M1'), Inverse(Matrix('M3')))

Example: Finding matches for a BLAS kernel
------------------------------------------

Lets assume we want to find all subexpressions of some expression which we can compute efficiently with
the `?TRMM`_ BLAS_ routine. These all have the form :math:`\alpha op(A) B` or :math:`\alpha B op(A)` where
:math:`op(A)` is either :math:`A` or :math:`A^T` and :math:`A` is a triangular matrix. Here, we will ignore
:math:`\alpha` and just assume it as 1.

First, we define the variables and constraints we need:

>>> A_ = Wildcard.symbol('A', Matrix)
>>> B_ = Wildcard.symbol('B', Matrix)
>>> before_ = Wildcard.star('before')
>>> after_ = Wildcard.star('after')
>>> A_is_triangular = CustomConstraint(lambda A: 'triangular' in A.properties)

Then we can construct the patterns, again using context variables to capture the remaining operands:

>>> trmm_patterns = [
...     Pattern(Times(before_, A_, B_, after_), A_is_triangular),
...     Pattern(Times(before_, Transpose(A_), B_, after_), A_is_triangular),
...     Pattern(Times(before_, B_, A_, after_), A_is_triangular),
...     Pattern(Times(before_, B_, Transpose(A_), after_), A_is_triangular),
... ]

Then, we can find all matching subexpressions using `.one_to_one.match`:

>>> expr = Times(Transpose(M3), M1, M3, M2)
>>> for i, pattern in enumerate(trmm_patterns):
...     for substitution in match(expr, pattern):
...         print('Pattern {} matched with {} as A and {} as B'.format(i, substitution['A'], substitution['B']))
Pattern 0 matched with M3 as A and M2 as B
Pattern 1 matched with M3 as A and M1 as B
Pattern 2 matched with M3 as A and M1 as B

.. _`?TRMM`: https://software.intel.com/en-us/node/468494
.. _BLAS: http://www.netlib.org/blas/
