# -*- coding: utf-8 -*-
from typing import (Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Type, Union, cast, Set)

from multiset import Multiset

from ..expressions.expressions import Expression, Operation, Symbol, SymbolWildcard, Wildcard
from ..expressions.constraints import Constraint
from ..expressions.substitution import Substitution
from ..utils import (
    VariableWithCount, commutative_sequence_variable_partition_iter, fixed_integer_vector_iter,
    integer_partition_vector_iter, generator_chain
)

__all__ = ['CommutativePatternsParts', 'Matcher']

Matcher = Callable[[Sequence[Expression], Expression, Substitution], Iterator[Substitution]]
VarInfo = NamedTuple('VarInfo', [('min_count', int), ('type', Optional[type])])
MultisetOfExpression = Multiset


class CommutativePatternsParts(object):
    """Representation of the parts of a commutative pattern expression.

    This data structure contains all the operands of a commutative operation pattern.
    They are distinguished by how they need to be matched against an expression.

    All parts are represented by a :class:`.Multiset`, because the order of operands does not matter
    in a commutative operation.

    In addition, some lengths are precalculated during the initialization of this data structure
    so that they do not have to be recalculated later.

    This data structure is meant to be immutable, so do not change any of its attributes!

    Attributes:
        operation (Type[Operation]):
            The type of of the original pattern expression. Must be a subclass of
            :class:`.Operation`.

        constant (MultisetOfExpression):
            A :class:`~.Multiset` representing the constant operands of the pattern.
            An expression is constant, if it does not contain variables or wildcards.
        syntactic (Multiset[Operation]):
            A :class:`.Multiset` representing the syntactic operands of the pattern.
            An expression is syntactic, if it does contain neither associative nor commutative operations
            nor sequence variables. Here, constant expressions and variables also get their own counters,
            so they are not included in this counter.
        sequence_variables (Multiset[str]):
            A :class:`.Multiset` representing the sequence variables of the pattern.
            Variables are represented by their name. Additional information is stored in
            ``sequence_variable_infos``. For wildcards without variable, the name will be ``None``.
        sequence_variable_infos (Dict[str, VarInfo]):
            A dictionary mapping sequence variable names to more information about the variable, i.e. its
            ``min_count`` and ``constraint``.
        fixed_variables (Multiset[VarInfo]):
            A :class:`.Multiset` representing the fixed length variables of the pattern.
            Here the key is a tuple of the form `(name, length)` of the variable.
            For wildcards without variable, the name will be `None`.
        fixed_variable_infos (Dict[str, VarInfo]):
            A dictionary mapping fixed variable names to more information about the variable, i.e. its
            ``min_count`` and ``constraint``.
        rest (MultisetOfExpression):
            A :class:`.Multiset` representing the operands of the pattern that do not fall
            into one of the previous categories. That means it contains operation expressions, which
            are not syntactic.

        length (int):
            The total count of operands of the commutative operation pattern.
        sequence_variable_min_length (int):
            The total combined minimum length of all sequence variables in the commutative
            operation pattern. This is the sum of the `min_count` attributes of the sequence
            variables.
        fixed_variable_length (int):
            The total combined length of all fixed length variables in the commutative
            operation pattern. This is the sum of the `min_count` attributes of the
            variables.
        wildcard_fixed (Optional[bool]):
            Iff none of the operands is an unnamed wildcards, it is ``None``. Iff there are any unnamed sequence wildcards, it is
            ``True``. Otherwise, it is ``False``.
        wildcard_min_length (int):
            If :attr:`wildcard_fixed` is not ``None``, this is the total combined minimum length of all unnamed
            wildcards.
    """

    def __init__(self, operation: Type[Operation], *expressions: Expression) -> None:
        """Create a CommutativePatternsParts instance.

        Args:
            operation:
                The type of the commutative operation. Must be a subclass of :class:`.Operation` with
                :attr:`~.Operation.commutative` set to ``True``.
            *expressions:
                The operands of the commutative operation.
        """
        self.operation = operation
        self.length = len(expressions)

        self.constant = Multiset()  # type: MultisetOfExpression
        self.syntactic = Multiset()  # type: MultisetOfExpression
        self.sequence_variables = Multiset()  # type: Multiset[str]
        self.sequence_variable_infos = dict()
        self.fixed_variables = Multiset()  # type: Multiset[str]
        self.fixed_variable_infos = dict()
        self.rest = Multiset()  # type: MultisetOfExpression

        self.sequence_variable_min_length = 0
        self.fixed_variable_length = 0
        self.wildcard_min_length = 0
        self.wildcard_fixed = None

        for expression in expressions:
            expression = expression
            if expression.is_constant:
                self.constant[expression] += 1
            elif expression.head is None:
                wc = cast(Wildcard, expression)
                if wc.variable:
                    name = wc.variable
                    if wc.fixed_size:
                        self.fixed_variables[name] += 1
                        symbol_type = getattr(wc, 'symbol_type', None)
                        self._update_var_info(self.fixed_variable_infos, name, wc.min_count, symbol_type)
                        self.fixed_variable_length += wc.min_count
                    else:
                        self.sequence_variables[name] += 1
                        self._update_var_info(self.sequence_variable_infos, name, wc.min_count)
                        self.sequence_variable_min_length += wc.min_count
                else:
                    self.wildcard_min_length += wc.min_count
                    if self.wildcard_fixed is None:
                        self.wildcard_fixed = wc.fixed_size
                    else:
                        self.wildcard_fixed = self.wildcard_fixed and wc.fixed_size
            elif expression.is_syntactic:
                self.syntactic[expression] += 1
            else:
                self.rest[expression] += 1

    @staticmethod
    def _update_var_info(infos, name, count, symbol_type=None):
        if name not in infos:
            infos[name] = VarInfo(count, symbol_type)
        else:
            existing_info = infos[name]
            assert existing_info.min_count == count
            assert existing_info.type == symbol_type

    def __str__(self):
        parts = []
        parts.extend(map(str, self.constant))
        parts.extend(map(str, self.syntactic))
        parts.extend(map(str, self.rest))

        for name, count in self.sequence_variables.items():
            parts.extend([name] * count)

        for name, count in self.fixed_variables.items():
            parts.extend([name] * count)

        return '{}({})'.format(self.operation.name, ', '.join(parts))


def _match(subjects: List[Expression], pattern: Expression, subst: Substitution, constraints: Set[Constraint]) -> Iterator[Substitution]:
    match_iter = None
    expr = subjects[0] if subjects else None
    if isinstance(pattern, Wildcard):
        match_iter = _match_wildcard(subjects, pattern, subst)
        if not pattern.fixed_size:
            expr = tuple(subjects)

    elif isinstance(pattern, Symbol):
        if len(subjects) == 1 and isinstance(subjects[0], type(pattern)) and subjects[0].name == pattern.name:
            match_iter = iter([subst])

    elif isinstance(pattern, Operation):
        if len(subjects) != 1 or not isinstance(subjects[0], pattern.__class__):
            return
        op_expr = cast(Operation, subjects[0])
        if not op_expr.symbols >= pattern.symbols:
            return
        match_iter = _match_operation(op_expr.operands, pattern, subst, _match, constraints)

    else:
        assert False, "Unexpected pattern of type {!r}".format(type(pattern))

    if pattern.variable:
        for new_subst in match_iter:
            try:
                new_subst = new_subst.union_with_variable(pattern.variable, expr)
            except ValueError:
                pass
            else:
                yield from check_constraints(new_subst, constraints)
    elif match_iter:
        yield from match_iter


def check_constraints(substitution, constraints):
    restore_constraints = set()
    for constraint in list(constraints):
        for var in constraint.variables:
            if var not in substitution:
                break
        else:
            restore_constraints.add(constraint)
            constraints.remove(constraint)
            if not constraint(substitution):
                break
    else:
        yield substitution
        for constraint in restore_constraints:
            constraints.add(constraint)


def _match_wildcard(subjects: List[Expression], wildcard: Wildcard, subst: Substitution) -> Iterator[Substitution]:
    if wildcard.fixed_size:
        if len(subjects) == wildcard.min_count:
            if isinstance(wildcard, SymbolWildcard) and not isinstance(subjects[0], wildcard.symbol_type):
                return
            yield subst
    elif len(subjects) >= wildcard.min_count:
        yield subst


def _match_factory(expressions, operand, constraints, matcher):
    def factory(subst):
        yield from matcher(expressions, operand, subst, constraints)

    return factory


def _count_seq_vars(expressions, operation):
    remaining = len(expressions)
    sequence_var_count = 0
    for operand in operation.operands:
        if isinstance(operand, Wildcard):
            if not operand.fixed_size or operation.associative:
                sequence_var_count += 1
            remaining -= operand.min_count
        else:
            remaining -= 1
        if remaining < 0:
            raise ValueError
    return remaining, sequence_var_count


def _build_full_partition(sequence_var_partition: Sequence[int], subjects: Sequence[Expression],
                          operation: Operation) -> List[Sequence[Expression]]:
    """Distribute subject operands among pattern operands.

    Given a partitoning for the variable part of the operands (i.e. a list of how many extra operands each sequence
    variable gets assigned).
    """
    i = 0
    var_index = 0
    result = []
    for operand in operation.operands:
        wrap_associative = False
        if isinstance(operand, Wildcard):
            count = operand.min_count
            if not operand.fixed_size or operation.associative:
                count += sequence_var_partition[var_index]
                var_index += 1
                wrap_associative = operand.fixed_size and operand.min_count
        else:
            count = 1

        operand_expressions = subjects[i:i + count]
        i += count

        if wrap_associative and len(operand_expressions) > wrap_associative:
            fixed = wrap_associative - 1
            operand_expressions = tuple(operand_expressions[:fixed]) + (type(operation)(*operand_expressions[fixed:]), )

        result.append(operand_expressions)

    return result


def _non_commutative_match(subjects, operation, subst, constraints, matcher):
    try:
        remaining, sequence_var_count = _count_seq_vars(subjects, operation)
    except ValueError:
        return
    for part in integer_partition_vector_iter(remaining, sequence_var_count):
        partition = _build_full_partition(part, subjects, operation)
        factories = [_match_factory(e, o, constraints, matcher) for e, o in zip(partition, operation.operands)]

        for new_subst in generator_chain(subst, *factories):
            yield new_subst


def _match_operation(expressions, operation, subst, matcher, constraints):
    if len(operation.operands) == 0:
        if len(expressions) == 0:
            yield subst
        return
    if not operation.commutative:
        yield from _non_commutative_match(expressions, operation, subst, constraints, matcher)
    else:
        parts = CommutativePatternsParts(type(operation), *operation.operands)
        yield from _match_commutative_operation(expressions, parts, subst, constraints, matcher)


def _match_commutative_operation(
        subject_operands: Iterable[Expression], pattern: CommutativePatternsParts, substitution: Substitution, constraints, matcher
) -> Iterator[Substitution]:
    subjects = Multiset(subject_operands)  # type: MultisetOfExpression
    if not pattern.constant <= subjects:
        return
    subjects -= pattern.constant
    rest_expr = pattern.rest + pattern.syntactic
    needed_length = (
        pattern.sequence_variable_min_length + pattern.fixed_variable_length + len(rest_expr) +
        pattern.wildcard_min_length
    )

    if len(subjects) < needed_length:
        return

    fixed_vars = Multiset(pattern.fixed_variables)  # type: Multiset[str]
    for name, count in pattern.fixed_variables.items():
        if name in substitution:
            replacement = substitution[name]
            if pattern.operation.associative and isinstance(replacement, pattern.operation):
                needed_count = Multiset(cast(Operation, substitution[name]).operands)  # type: MultisetOfExpression
            else:
                if not isinstance(replacement, Expression):
                    return
                needed_count = Multiset({replacement: 1})
            if count > 1:
                needed_count *= count
            if not needed_count <= subjects:
                return
            subjects -= needed_count
            del fixed_vars[name]

    factories = [_fixed_expr_factory(e, constraints, matcher) for e in rest_expr]

    if not pattern.operation.associative:
        for name, count in fixed_vars.items():
            min_count, symbol_type = pattern.fixed_variable_infos[name]
            factory = _fixed_var_iter_factory(name, count, min_count, symbol_type, constraints)
            factories.append(factory)

        if pattern.wildcard_fixed is True:
            factory = _fixed_var_iter_factory(None, 1, pattern.wildcard_min_length, None, constraints)
            factories.append(factory)
    else:
        for name, count in fixed_vars.items():
            min_count, symbol_type = pattern.fixed_variable_infos[name]
            if symbol_type is not None:
                factory = _fixed_var_iter_factory(name, count, min_count, symbol_type, constraints)
                factories.append(factory)

    expr_counter = Multiset(subjects)  # type: MultisetOfExpression

    for rem_expr, substitution in generator_chain((expr_counter, substitution), *factories):
        sequence_vars = _variables_with_counts(pattern.sequence_variables, pattern.sequence_variable_infos)
        if pattern.operation.associative:
            sequence_vars += _variables_with_counts(fixed_vars, pattern.fixed_variable_infos)
            if pattern.wildcard_fixed is True:
                sequence_vars += (VariableWithCount(None, 1, pattern.wildcard_min_length), )
        if pattern.wildcard_fixed is False:
            sequence_vars += (VariableWithCount(None, 1, pattern.wildcard_min_length), )

        for sequence_subst in commutative_sequence_variable_partition_iter(Multiset(rem_expr), sequence_vars):
            if pattern.operation.associative:
                for v in fixed_vars.distinct_elements():
                    if v not in sequence_subst:
                        continue
                    l = pattern.fixed_variable_infos[v].min_count
                    value = cast(MultisetOfExpression, sequence_subst[v])
                    if len(value) > l:
                        normal = Multiset(list(value)[:l - 1])
                        wrapped = pattern.operation(*(value - normal))
                        normal.add(wrapped)
                        sequence_subst[v] = normal if l > 1 else next(iter(normal))
                    else:
                        assert len(value) == 1 and l == 1, "Fixed variables with length != 1 are not supported."
                        sequence_subst[v] = next(iter(value))
            try:
                result = substitution.union(sequence_subst)
            except ValueError:
                pass
            else:
                yield from check_constraints(result, constraints)


def _variables_with_counts(variables, infos):
    return tuple(
        VariableWithCount(name, count, infos[name].min_count)
        for name, count in variables.items() if infos[name].type is None
    )


def _fixed_expr_factory(expression, constraints, matcher):
    def factory(data):
        expressions, substitution = data
        for expr in expressions.distinct_elements():
            if expr.head == expression.head:
                for subst in matcher([expr], expression, substitution, constraints):
                    yield expressions - Multiset({expr: 1}), subst

    return factory


def _fixed_var_iter_factory(variable, count, length, symbol_type, constraints):
    def factory(data):
        expressions, substitution = data
        if variable in substitution:
            value = ([substitution[variable]]
                     if isinstance(substitution[variable], Expression) else substitution[variable])
            existing = Multiset(value) * count
            if not existing <= expressions:
                return
            yield expressions - existing, substitution
        else:
            if length == 1:
                for expr, expr_count in expressions.items():
                    if expr_count >= count and (symbol_type is None or isinstance(expr, symbol_type)):
                        if variable is not None:
                            new_substitution = Substitution(substitution)
                            new_substitution[variable] = expr
                            for new_substitution in check_constraints(new_substitution, constraints):
                                yield expressions - Multiset({expr: count}), new_substitution
                        else:
                            yield expressions - Multiset({expr: count}), substitution
            else:
                assert variable is None, "Fixed variables with length != 1 are not supported."
                exprs_with_counts = list(expressions.items())
                counts = tuple(c // count for _, c in exprs_with_counts)
                for subset in fixed_integer_vector_iter(counts, length):
                    sub_counter = Multiset(dict((exprs_with_counts[i][0], c * count) for i, c in enumerate(subset)))
                    yield expressions - sub_counter, substitution

    return factory
