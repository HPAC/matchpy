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

>>> A = Matrix('A', ['diagonal', 'square'])
>>> B = Matrix('B', ['symmetric', 'square'])

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

>>> print(Plus(Times(v, Transpose(v)), Times(a, Inverse(A))))
((a * I(A)) + (v * (v)^T))

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

Patterns can be limited in what is matched by adding constraints. A constraints is essentially a callback,
that gets the match substitution and can return either ``True`` or ``False``. You can either use the `.CustomConstraint`
class with any (lambda) function, or create your own subclass of `.Constraint`.

For example, if we want to only match diagonal matrices with a certain variable, we can create a constraint for that:

>>> C_ = Wildcard.symbol('C', Matrix)
>>> C_is_diagonal_matrix = CustomConstraint(lambda C: 'diagonal' in C.properties)
>>> pattern = Pattern(C_, C_is_diagonal_matrix)

Then the variable *C* will only match diagonal matrices:

>>> is_match(A, pattern)
True
>>> is_match(B, pattern)
False
