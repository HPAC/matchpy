# -*- coding: utf-8 -*-
from typing import (Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Type, Union, cast)

from multiset import Multiset

from ..expressions import (
    Expression, FrozenExpression, Operation, Substitution, Symbol, SymbolWildcard, Variable, Wildcard, Constraint,
    MultiConstraint
)
from ..utils import (
    VariableWithCount, commutative_sequence_variable_partition_iter, fixed_integer_vector_iter,
    integer_partition_vector_iter, generator_chain
)

__all__ = ['CommutativePatternsParts', 'Matcher']

Matcher = Callable[[Sequence[FrozenExpression], FrozenExpression, Substitution], Iterator[Substitution]]
VarInfo = NamedTuple('VarInfo', [('min_count', int), ('constraint', Constraint), ('type', Optional[type])])


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

        constant (Multiset[Expression]):
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
        rest (Multiset[Expression]):
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
            Iff none of the operands is an unnamed wildcards (i.e. a :class:`.Wildcard` not wrapped in as
            :class:`.Variable`), it is ``None``. Iff there are any unnamed sequence wildcards, it is
            ``True``. Otherwise, it is ``False``.
        wildcard_min_length (int):
            If :attr:`wildcard_fixed` is not ``None``, this is the total combined minimum length of all unnamed
            wildcards.
    """

    def __init__(self, operation: Type[Operation], *expressions: FrozenExpression) -> None:
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

        self.constant = Multiset()  # type: Multiset[Expression]
        self.syntactic = Multiset()  # type: Multiset[Expression]
        self.sequence_variables = Multiset()  # type: Multiset[str]
        self.sequence_variable_infos = dict()
        self.fixed_variables = Multiset()  # type: Multiset[str]
        self.fixed_variable_infos = dict()
        self.rest = Multiset()  # type: Multiset[Expression]

        self.sequence_variable_min_length = 0
        self.fixed_variable_length = 0
        self.wildcard_min_length = 0
        self.wildcard_fixed = None

        for expression in expressions:
            if expression.is_constant:
                self.constant[expression] += 1
            elif expression.head is None:
                wc = cast(Wildcard, expression)
                constraint = wc.constraint
                if isinstance(wc, Variable):
                    name = wc.name
                    wc = cast(Wildcard, wc.expression)
                    if wc.fixed_size:
                        self.fixed_variables[name] += 1
                        symbol_type = getattr(wc, 'symbol_type', None)
                        self._update_var_info(self.fixed_variable_infos, name, wc.min_count, constraint, symbol_type)
                        self.fixed_variable_length += wc.min_count
                    else:
                        self.sequence_variables[name] += 1
                        self._update_var_info(self.sequence_variable_infos, name, wc.min_count, constraint)
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
    def _update_var_info(infos, name, count, constraint, symbol_type=None):
        if name not in infos:
            infos[name] = VarInfo(count, constraint, symbol_type)
        else:
            existing_info = infos[name]
            assert existing_info.min_count == count
            assert existing_info.type == symbol_type
            if constraint is not None:
                assert name is not None
                if existing_info.constraint is not None:
                    constraint = MultiConstraint.create(existing_info.constraint, constraint)
                infos[name] = VarInfo(count, constraint, symbol_type)

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


def _match(subjects: List[Expression], pattern: Expression, subst: Substitution) -> Iterator[Substitution]:
    if isinstance(pattern, Variable):
        yield from _match_variable(subjects, pattern, subst, _match)

    elif isinstance(pattern, Wildcard):
        yield from _match_wildcard(subjects, pattern, subst)

    elif isinstance(pattern, Symbol):
        if len(subjects) == 1 and isinstance(subjects[0], type(pattern)) and subjects[0].name == pattern.name:
            if pattern.constraint is None or pattern.constraint(subst):
                yield subst

    else:
        assert isinstance(pattern, Operation), "Unexpected expression of type {!r}".format(type(pattern))
        if len(subjects) != 1 or not isinstance(subjects[0], pattern.__class__):
            return
        op_expr = cast(Operation, subjects[0])
        if not op_expr.symbols >= pattern.symbols:
            return
        for result in _match_operation(op_expr.operands, pattern, subst, _match):
            if pattern.constraint is None or pattern.constraint(result):
                yield result


def _match_variable(subjects: List[Expression], variable: Variable, subst: Substitution,
                    matcher: Matcher) -> Iterator[Substitution]:
    inner = variable.expression
    if len(subjects) == 1 and (not isinstance(inner, Wildcard) or inner.fixed_size):
        expr = next(iter(subjects))  # type: Union[Expression, List[Expression]]
    else:
        expr = tuple(subjects)
    for new_subst in matcher(subjects, inner, subst):
        try:
            new_subst = new_subst.union_with_variable(variable.name, expr)
            if variable.constraint is None or variable.constraint(new_subst):
                yield new_subst
        except ValueError:
            pass


def _match_wildcard(subjects: List[Expression], wildcard: Wildcard, subst: Substitution) -> Iterator[Substitution]:
    if wildcard.fixed_size:
        if len(subjects) == wildcard.min_count:
            if isinstance(wildcard, SymbolWildcard) and not isinstance(subjects[0], wildcard.symbol_type):
                return
            if wildcard.constraint is None or wildcard.constraint(subst):
                yield subst
    elif len(subjects) >= wildcard.min_count:
        if wildcard.constraint is None or wildcard.constraint(subst):
            yield subst


def _match_factory(expressions, operand, matcher):
    def factory(subst):
        yield from matcher(expressions, operand, subst)

    return factory


def _count_seq_vars(expressions, operation):
    remaining = len(expressions)
    sequence_var_count = 0
    for operand in operation.operands:
        if isinstance(operand, Variable):
            operand = operand.expression
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
        inner = operand.expression if isinstance(operand, Variable) else operand
        if isinstance(inner, Wildcard):
            count = inner.min_count
            if not inner.fixed_size or operation.associative:
                count += sequence_var_partition[var_index]
                var_index += 1
                wrap_associative = inner.fixed_size and inner.min_count
        else:
            count = 1

        operand_expressions = subjects[i:i + count]
        i += count

        if wrap_associative and len(operand_expressions) > wrap_associative:
            fixed = wrap_associative - 1
            op_factory = type(operation).from_args
            operand_expressions = tuple(operand_expressions[:fixed]) + (op_factory(*operand_expressions[fixed:]), )

        result.append(operand_expressions)

    return result


def _non_commutative_match(subjects, operation, subst, matcher):
    try:
        remaining, sequence_var_count = _count_seq_vars(subjects, operation)
    except ValueError:
        return
    for part in integer_partition_vector_iter(remaining, sequence_var_count):
        partition = _build_full_partition(part, subjects, operation)
        factories = [_match_factory(e, o, matcher) for e, o in zip(partition, operation.operands)]

        for new_subst in generator_chain(subst, *factories):
            yield new_subst


def _match_operation(expressions, operation, subst, matcher):
    if len(operation.operands) == 0:
        if len(expressions) == 0:
            yield subst
        return
    if not operation.commutative:
        yield from _non_commutative_match(expressions, operation, subst, matcher)
    else:
        parts = CommutativePatternsParts(type(operation), *operation.operands)
        yield from _match_commutative_operation(expressions, parts, subst, matcher)


def _match_commutative_operation(
        subject_operands: Iterable[Expression],
        pattern: CommutativePatternsParts,
        substitution: Substitution,
        matcher,
        syntactic_matcher=None
) -> Iterator[Substitution]:
    if any(not e.is_constant for e in subject_operands):
        raise ValueError("All given expressions must be constant.")
    expressions = Multiset(subject_operands)  # type: Multiset[Expression]
    if not pattern.constant <= expressions:
        return
    expressions -= pattern.constant
    if syntactic_matcher is not None and pattern.syntactic:
        rest, syntactics = _split_expressions(expressions)
        if len(pattern.syntactic) > len(syntactics):
            return
        for subst, remaining in syntactic_matcher(syntactics, pattern.syntactic):
            try:
                subst = subst.union(substitution)
                yield from _matches_from_matching(subst, remaining + rest, pattern, matcher, False)
            except ValueError:
                pass
    else:
        yield from _matches_from_matching(substitution, expressions, pattern, matcher, True)


def _matches_from_matching(
        subst: Substitution, remaining: Multiset, pattern: CommutativePatternsParts, matcher, include_syntactic: bool
) -> Iterator[Substitution]:
    rest_expr = (pattern.rest + pattern.syntactic) if include_syntactic else pattern.rest
    needed_length = (
        pattern.sequence_variable_min_length + pattern.fixed_variable_length + len(rest_expr) +
        pattern.wildcard_min_length
    )

    if len(remaining) < needed_length:
        return

    fixed_vars = Multiset(pattern.fixed_variables)  # type: Multiset[str]
    for name, count in pattern.fixed_variables.items():
        if name in subst:
            if pattern.operation.associative and isinstance(subst[name], pattern.operation):
                needed_count = Multiset(cast(Operation, subst[name]).operands)  # type: Multiset[Expression]
            elif isinstance(subst[name], Expression):
                needed_count = Multiset({subst[name]: 1})
            else:
                needed_count = Multiset(cast(Iterable[Expression], subst[name]))
            if count > 1:
                needed_count *= count
            if not needed_count <= remaining:
                return
            remaining -= needed_count
            del fixed_vars[name]

    factories = [_fixed_expr_factory(e, matcher) for e in rest_expr]

    if not pattern.operation.associative:
        for name, count in fixed_vars.items():
            info = pattern.fixed_variable_infos[name]
            factory = _fixed_var_iter_factory(name, count, info.min_count, info.constraint, info.type)
            factories.append(factory)

        if pattern.wildcard_fixed is True:
            factory = _fixed_var_iter_factory(None, 1, pattern.wildcard_min_length, None)
            factories.append(factory)
    else:
        for name, count in fixed_vars.items():
            min_count, constraint, symbol_type = pattern.fixed_variable_infos[name]
            if symbol_type is not None:
                factory = _fixed_var_iter_factory(name, count, min_count, constraint, symbol_type)
                factories.append(factory)

    expr_counter = Multiset(remaining)  # type: Multiset[Expression]

    for rem_expr, subst in generator_chain((expr_counter, subst), *factories):
        sequence_vars = _variables_with_counts(pattern.sequence_variables, pattern.sequence_variable_infos)
        constraints = [pattern.sequence_variable_infos[name].constraint for name in pattern.sequence_variables]
        if pattern.operation.associative:
            sequence_vars += _variables_with_counts(fixed_vars, pattern.fixed_variable_infos)
            constraints += [pattern.fixed_variable_infos[name].constraint for name in fixed_vars]
            if pattern.wildcard_fixed is True:
                sequence_vars += (VariableWithCount(None, 1, pattern.wildcard_min_length), )
        if pattern.wildcard_fixed is False:
            sequence_vars += (VariableWithCount(None, 1, pattern.wildcard_min_length), )
        combined_constraint = MultiConstraint.create(*constraints)

        for sequence_subst in commutative_sequence_variable_partition_iter(Multiset(rem_expr), sequence_vars):
            if pattern.operation.associative:
                for v in fixed_vars.keys():
                    if v not in sequence_subst:
                        continue
                    l = pattern.fixed_variable_infos[v].min_count
                    value = cast(Multiset[Expression], sequence_subst[v])
                    if len(value) > l:
                        normal = Multiset(list(value)[:l - 1])
                        wrapped = pattern.operation.from_args(*(value - normal))
                        normal.add(wrapped)
                        sequence_subst[v] = normal if l > 1 else next(iter(normal))
                    elif l == len(value) and l == 1:
                        sequence_subst[v] = next(iter(value))
            try:
                result = subst.union(sequence_subst)
                if combined_constraint is None or combined_constraint(result):
                    yield result
            except ValueError:
                pass


def _variables_with_counts(variables, infos):
    return tuple(
        VariableWithCount(name, count, infos[name].min_count)
        for name, count in variables.items() if infos[name].type is None
    )


def _fixed_expr_factory(expression, matcher):
    def factory(data):
        expressions, substitution = data
        for expr in expressions.keys():
            if expr.head == expression.head:
                for subst in matcher([expr], expression, substitution):
                    if expression.constraint is None or expression.constraint(subst):
                        yield expressions - Multiset({expr: 1}), subst

    return factory


def _fixed_var_iter_factory(variable, count, length, constraint=None, symbol_type=None):
    def factory(data):
        expressions, substitution = data
        if variable in substitution:
            value = ([substitution[variable]]
                     if isinstance(substitution[variable], Expression) else substitution[variable])
            existing = Multiset(value) * count
            if not existing <= expressions:
                return
            if constraint is None or constraint(substitution):
                yield expressions - existing, substitution
        else:
            if length == 1:
                for expr, expr_count in expressions.items():
                    if expr_count >= count and (symbol_type is None or isinstance(expr, symbol_type)):
                        if variable is not None:
                            new_substitution = Substitution(substitution)
                            new_substitution[variable] = expr
                            if constraint is None or constraint(new_substitution):
                                yield expressions - Multiset({expr: count}), new_substitution
                        else:
                            yield expressions - Multiset({expr: count}), substitution
            else:
                exprs_with_counts = list(expressions.items())
                counts = tuple(c // count for _, c in exprs_with_counts)
                for subset in fixed_integer_vector_iter(counts, length):
                    sub_counter = Multiset(dict((exprs_with_counts[i][0], c * count) for i, c in enumerate(subset)))
                    if variable is not None:
                        new_substitution = Substitution(substitution)
                        new_substitution[variable] = sub_counter
                        if constraint is None or constraint(new_substitution):
                            yield expressions - sub_counter, new_substitution
                    else:
                        yield expressions - sub_counter, substitution

    return factory


def _split_expressions(expressions: Multiset[Expression]) -> Tuple[Multiset[Expression], Multiset[Expression]]:
    constants = Multiset()  # type: Multiset[Expression]
    syntactics = Multiset()  # type: Multiset[Expression]

    for expression, count in expressions.items():
        if expression.is_syntactic or not (
                isinstance(expression, Operation) and (expression.associative or expression.commutative)
        ):
            syntactics[expression] = count
        else:
            constants[expression] = count

    return constants, syntactics
