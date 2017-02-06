********
Glossary
********

.. glossary::

    syntactic
        An :class:`.Expression` is syntactic iff it contains neither associative nor commutative operations and also
        does not contain sequence :class:`wildcards <.Wildcard>` (i.e. :class:`wildcards <.Wildcard>` with
        :attr:`~.Wildcard.fixed_size` set to ``False``).

    constant
        An :class:`.Expression` is :attr:`constant <.Expression.is_constant>` iff it does not contain any
        :class:`.Wildcard`.