import re

from ..expressions.expressions import Wildcard, AssociativeOperation, SymbolWildcard
from ..expressions.constraints import CustomConstraint
from ..expressions.functions import op_iter, get_variables
from .syntactic import OPERATION_END, is_operation
from .many_to_one import _EPS
from ..utils import get_short_lambda_source

COLLAPSE_IF_RE = re.compile(
    r'\n(?P<indent1>\s*)if (?P<cond1>[^\n]+):\n+\1(?P<indent2>\s+)'
    r'(?P<comment>(?:\#[^\n]*\n+\1\3)*)'
    r'if (?P<cond2>[^\n]+):\n+'
    r'(?P<block>\1\3(?P<indent3>\s+)[^\n]*\n+(?:\1\3\7[^\n]*\n+)*)'
    r'(?!\1(?:\3|elif|else))'
)


class CodeGenerator:
    def __init__(self, matcher):
        self._matcher = matcher
        self._var_number = 0
        self._indentation = '    '
        self._level = 0
        self._code = ''
        self._subjects = ['subjects']
        self._substs = 0
        self._patterns = set(range(len(matcher.patterns)))
        self._associative = 0
        self._associative_stack = [None]
        self._global_code = []
        self._imports = set()

    def indent(self):
        self._level += 1

    def dedent(self):
        if self._level > 0:
            self._level -= 1

    def add_line(self, line):
        self._code += (self._indentation * self._level) + str(line) + '\n'

    def get_var_name(self, prefix):
        self._var_number += 1
        return prefix + str(self._var_number)

    def generate_code(self, func_name='match_root', add_imports=True):
        self._imports.add('from collections import deque')
        self.add_line('def {}(subject):'.format(func_name))
        self.indent()
        self.add_line('{} = deque([subject]) if subject is not None else deque()'.format(self._subjects[-1]))
        self.add_line('subst{} = Substitution()'.format(self._substs))
        self.generate_state_code(self._matcher.root)
        self.add_line('return')
        self.add_line('yield')
        self.dedent()

        if add_imports:
            self._global_code.insert(0, '\n'.join(self._imports))

        return self.clean_code('\n\n'.join(p for p in self._global_code if p)), self.clean_code(self._code)

    def final_label(self, index, subst_name):
        return str(index)

    def generate_state_code(self, state):
        if state.matcher is not None:
            self._imports.add('from matchpy.matching.many_to_one import CommutativeMatcher')
            self._imports.add('from multiset import Multiset')
            self._imports.add('from matchpy.utils import VariableWithCount')
            generator = type(self)(state.matcher.automaton)
            generator.indent()
            global_code, code = generator.generate_code(func_name='get_match_iter', add_imports=False)
            self._global_code.append(global_code)
            patterns = self.commutative_patterns(state.matcher.patterns)
            subjects = repr(state.matcher.subjects)
            subjects_by_id = repr(state.matcher.subjects_by_id)
            associative = self.operation_symbol(state.matcher.associative)
            max_optional_count = repr(state.matcher.max_optional_count)
            anonymous_patterns = repr(state.matcher.anonymous_patterns)
            self._global_code.append(
                '''
class CommutativeMatcher{0}(CommutativeMatcher):
{8}_instance = None
{8}patterns = {1}
{8}subjects = {2}
{8}subjects_by_id = {7}
{8}bipartite = BipartiteGraph()
{8}associative = {3}
{8}max_optional_count = {4}
{8}anonymous_patterns = {5}

{8}def __init__(self):
{8}{8}self.add_subject(None)

{8}@staticmethod
{8}def get():
{8}{8}if CommutativeMatcher{0}._instance is None:
{8}{8}{8}CommutativeMatcher{0}._instance = CommutativeMatcher{0}()
{8}{8}return CommutativeMatcher{0}._instance

{8}@staticmethod
{6}'''.strip().format(
                    state.number, patterns, subjects, associative, max_optional_count, anonymous_patterns, code,
                    subjects_by_id, self._indentation
                )
            )
            self.add_line('matcher = CommutativeMatcher{}.get()'.format(state.number))
            tmp = self.get_var_name('tmp')
            self.add_line('{} = {}'.format(tmp, self._subjects[-1]))
            self.add_line('{} = []'.format(self._subjects[-1]))
            self.add_line('for s in {}:'.format(tmp))
            self.indent()
            self.add_line('matcher.add_subject(s)')
            subjects = self._subjects.pop()
            self.dedent()
            self.add_line(
                'for pattern_index, subst{} in matcher.match({}, subst{}):'.format(self._substs + 1, tmp, self._substs)
            )
            self._substs += 1
            self.indent()
            self.add_line('pass')
            for pattern_index, transitions in state.transitions.items():
                self.add_line('if pattern_index == {}:'.format(pattern_index))
                self.indent()
                self.add_line('pass')
                patterns, variables = next((p, v) for i, p, v in state.matcher.patterns.values() if i == pattern_index)
                variables = set(v[0][0] for v in variables)
                pvars = iter(get_variables(state.matcher.automaton.patterns[i][0].expression) for i in patterns)
                variables.update(*pvars)
                constraints = []
                if variables:
                    constraints = sorted(
                        set.union(*iter(self._matcher.constraint_vars.get(v, set()) for v in variables))
                    )
                self.generate_constraints(constraints, transitions)
                self.dedent()
            self.dedent()
            self._substs -= 1
            self._subjects.append(subjects)
        else:
            self.add_line('# State {}'.format(state.number))
            if state.number in self._matcher.finals:
                self.add_line('if len({}) == 0:'.format(self._subjects[-1]))
                self.indent()
                self.add_line('pass')
                for pattern_index in self._patterns:
                    constraints = self._matcher.patterns[pattern_index][0].global_constraints
                    for constraint in constraints:
                        self.enter_global_constraint(constraint)
                    self.yield_final_substitution(pattern_index)
                    for constraint in constraints:
                        self.exit_global_constraint(constraint)
                self.dedent()
            else:
                for transitions in state.transitions.values():
                    for transition in transitions:
                        self.generate_transition_code(transition)

    def commutative_var_entry(self, entry):
        return '(VariableWithCount({!r}, {}, {}, {}), {})'.format(
            entry[0][0], entry[0][1], entry[0][2],
            self.expr(entry[0][3]), self.operation_symbol(entry[1]) if isinstance(entry[1], type) else repr(entry[1])
        )

    def commutative_patterns(self, patterns):
        patterns = sorted(patterns.values(), key=lambda x: x[0])
        return '{{\n    {}\n}}'.format(
            ',\n    '.join(
                '{0}: ({0}, {1!r}, [\n      {2}\n])'.format(i, s, ',\n      '.join(map(self.commutative_var_entry, v)))
                for i, s, v in patterns
            )
        )

    def generate_transition_code(self, transition):
        enter_func = None
        exit_func = None
        if is_operation(transition.label):
            enter_func = self.enter_operation
            exit_func = self.exit_operation
        elif transition.label == _EPS:
            enter_func = self.enter_eps
            exit_func = self.exit_eps
        elif isinstance(transition.label, Wildcard):
            wc = transition.label
            if wc.optional is not None:
                self.enter_variable_assignment(transition.variable_name, self.optional_expr(wc.optional))
                constraints = sorted(transition.check_constraints) if transition.check_constraints is not None else []
                self.generate_constraints(constraints, [transition])
                self.exit_variable_assignment()
            if isinstance(wc, SymbolWildcard):
                enter_func = self.enter_symbol_wildcard
                exit_func = self.exit_symbol_wildcard
            elif wc.fixed_size and self._associative_stack[-1] is None:
                enter_func = self.enter_fixed_wildcard
                exit_func = self.exit_fixed_wildcard
            else:
                enter_func = self.enter_sequence_wildcard
                exit_func = self.exit_sequence_wildcard
        elif transition.label is OPERATION_END:
            enter_func = self.enter_operation_end
            exit_func = self.exit_operation_end
        else:
            enter_func = self.enter_symbol
            exit_func = self.exit_symbol

        value = enter_func(transition.label)
        value, var_value = value if isinstance(value, tuple) else (value, value)
        if transition.variable_name is not None:
            self.enter_variable_assignment(transition.variable_name, var_value)
        if transition.subst is not None:
            self.enter_subst(transition.subst)
        constraints = sorted(transition.check_constraints) if transition.check_constraints is not None else []
        self.generate_constraints(constraints, [transition])

        if transition.subst is not None:
            self.exit_subst(transition.subst)
        if transition.variable_name is not None:
            self.exit_variable_assignment()
        exit_func(value)

    def get_args(self, operation, operation_type):
        return 'op_iter({})'.format(operation)

    def push_subjects(self, value, operation):
        self._subjects.append(self.get_var_name('subjects'))
        self.add_line('{} = deque({})'.format(self._subjects[-1], self.get_args(value, operation)))

    def push_subst(self):
        new_subst = self.get_var_name('subst')
        self.add_line('subst{} = Substitution(subst{})'.format(self._substs + 1, self._substs))
        self._substs += 1

    def enter_eps(self, _):
        return '{0}'.format(self._subjects[-1])

    def exit_eps(self, _):
        pass

    def enter_operation(self, operation):
        self.add_line(
            'if len({0}) >= 1 and isinstance({0}[0], {1}):'.
            format(self._subjects[-1], self.operation_symbol(operation))
        )
        self.indent()
        tmp = self.get_var_name('tmp')
        self.add_line('{} = {}.popleft()'.format(tmp, self._subjects[-1]))
        atype = operation if issubclass(operation, AssociativeOperation) else None
        self._associative_stack.append(atype)
        if atype is not None:
            self._associative += 1
            self.add_line('associative{} = {}'.format(self._associative, tmp))
            self.add_line('associative_type{} = type({})'.format(self._associative, tmp))
        self.push_subjects(tmp, operation)
        return tmp

    def operation_symbol(self, operation):
        if operation is None:
            return 'None'
        return operation.__name__

    def exit_operation(self, value):
        self._subjects.pop()
        self.add_line('{}.appendleft({})'.format(self._subjects[-1], value))
        self.dedent()
        atype = self._associative_stack.pop()
        if atype is not None:
            self._associative -= 1

    def enter_symbol_wildcard(self, wildcard):
        self.add_line(
            'if len({0}) >= 1 and isinstance({0}[0], {1}):'.
            format(self._subjects[-1], self.symbol_type(wildcard.symbol_type))
        )
        self.indent()
        tmp = self.get_var_name('tmp')
        self.add_line('{} = {}.popleft()'.format(tmp, self._subjects[-1]))
        return tmp

    def symbol_type(self, symbol):
        return symbol.__name__

    def exit_symbol_wildcard(self, value):
        self.add_line('{}.appendleft({})'.format(self._subjects[-1], value))
        self.dedent()

    def enter_fixed_wildcard(self, wildcard):
        self.add_line('if len({}) >= 1:'.format(self._subjects[-1]))
        self.indent()
        tmp = self.get_var_name('tmp')
        self.add_line('{} = {}.popleft()'.format(tmp, self._subjects[-1]))
        return tmp

    def exit_fixed_wildcard(self, value):
        self.add_line('{}.appendleft({})'.format(self._subjects[-1], value))
        self.dedent()

    def enter_variable_assignment(self, variable_name, value):
        self.push_subst()
        self.add_line('try:')
        self.indent()
        self.add_line('subst{}.try_add_variable({!r}, {})'.format(self._substs, variable_name, value))
        self.dedent()
        self.add_line('except ValueError:')
        self.indent()
        self.add_line('pass')
        self.dedent()
        self.add_line('else:')
        self.indent()
        self.add_line('pass')

    def enter_subst(self, subst):
        self.push_subst()
        self.add_line('try:')
        self.indent()
        for name, value in subst.items():
            self.add_line('subst{}.try_add_variable({!r}, {})'.format(self._substs, name, self.expr(value)))
        self.dedent()
        self.add_line('except ValueError:')
        self.indent()
        self.add_line('pass')
        self.dedent()
        self.add_line('else:')
        self.indent()
        self.add_line('pass')

    def expr(self, expr):
        return repr(expr)

    def exit_subst(self, subst):
        self._substs -= 1
        self.dedent()

    def exit_variable_assignment(self):
        self._substs -= 1
        self.dedent()

    def enter_optional_wildcard(self, wildcard, variable_name):
        self.enter_variable_assignment(variable_name, self.optional_expr(wildcard.optional))

    def optional_expr(self, expr):
        return repr(expr)

    def exit_optional_wildcard(self, value):
        self.exit_variable_assignment()

    def enter_symbol(self, symbol):
        self.add_line('if len({0}) >= 1 and {0}[0] == {1}:'.format(self._subjects[-1], self.symbol_repr(symbol)))
        self.indent()
        tmp = self.get_var_name('tmp')
        self.add_line('{} = {}.popleft()'.format(tmp, self._subjects[-1]))
        return tmp

    def symbol_repr(self, symbol):
        return repr(symbol)

    def exit_symbol(self, value):
        self.add_line('{}.appendleft({})'.format(self._subjects[-1], value))
        self.dedent()

    def enter_operation_end(self, _):
        self.add_line('if len({0}) == 0:'.format(self._subjects[-1]))
        self.indent()
        self.add_line('pass')
        subjects = self._subjects.pop()
        atype = self._associative_stack.pop()
        if atype is not None:
            self._associative -= 1
        return [subjects, atype]

    def exit_operation_end(self, value):
        subjects, atype = value
        self._subjects.append(subjects)
        self.dedent()
        self._associative_stack.append(atype)
        if atype is not None:
            self._associative += 1

    def enter_sequence_wildcard(self, wildcard):
        tmp = self.get_var_name('tmp')
        tmp2 = self.get_var_name('tmp')
        mc = wildcard.min_count if wildcard.optional is None or wildcard.min_count > 0 else 1
        self.add_line('if len({}) >= {}:'.format(self._subjects[-1], mc))
        self.indent()
        self.add_line('{} = []'.format(tmp))
        for _ in range(mc):
            self.add_line('{}.append({}.popleft())'.format(tmp, self._subjects[-1]))
        self.add_line('while True:')
        self.indent()
        if self._associative_stack[-1] is not None and wildcard.fixed_size:
            self.add_line('if len({}) > {}:'.format(tmp, wildcard.min_count))
            self.indent()
            self.add_line(
                '{} = {}'.format(
                    tmp2,
                    self.create_operation(
                        'associative{}'.format(self._associative), 'associative{}'.format(self._associative), tmp
                    )
                )
            )
            self.dedent()
            self.add_line('elif len({}) == 1:'.format(tmp))
            self.indent()
            self.add_line('{} = {}[0]'.format(tmp2, tmp))
            self.dedent()
            self.add_line('else:')
            self.indent()
            self.add_line('assert False, "Unreachable"')
            self.dedent()
        else:
            self.add_line('{} = tuple({})'.format(tmp2, tmp))
        return tmp, tmp2

    def create_operation(self, operation, operation_type, args):
        return 'create_operation_expression({}, {})'.format(operation, args)

    def exit_sequence_wildcard(self, value):
        self.add_line('if len({}) == 0:'.format(self._subjects[-1]))
        self.indent()
        self.add_line('break')
        self.dedent()
        self.add_line('{}.append({}.popleft())'.format(value, self._subjects[-1]))
        self.dedent()
        self.add_line('{}.extendleft(reversed({}))'.format(self._subjects[-1], value))
        self.dedent()

    def yield_final_substitution(self, pattern_index):
        renaming = self._matcher.pattern_vars[pattern_index]
        subst_name = 'subst{}'.format(self._substs)
        if any(k != v for k, v in renaming.items()):
            self.add_line('tmp_subst = Substitution()')
            for original, renamed in renaming.items():
                self.add_line('tmp_subst[{!r}] = subst{}[{!r}]'.format(original, self._substs, renamed))
            subst_name = 'tmp_subst'
        self.add_line('# {}: {}'.format(pattern_index, self._matcher.patterns[pattern_index][0]))
        self.add_line('yield {}, {}'.format(self.final_label(pattern_index, subst_name), subst_name))

    def generate_constraints(self, constraints, transitions):
        if len(constraints) == 0:
            for transition in transitions:
                removed = self._patterns - transition.patterns
                self._patterns.intersection_update(transition.patterns)
                self.generate_state_code(transition.target)
                self._patterns.update(removed)
        else:
            constraint_index, *remaining = constraints
            constraint, patterns = self._matcher.constraints[constraint_index]
            remaining_patterns = self._patterns - patterns
            remaining_transitions = [t for t in transitions if t.patterns & remaining_patterns]
            checked_patterns = self._patterns & patterns
            checked_transitions = [t for t in transitions if t.patterns & checked_patterns]
            if checked_patterns and checked_transitions:
                cvars = ' or '.join('{!r} not in subst{}'.format(v, self._substs) for v in constraint.variables)
                if cvars:
                    cvars += ' or '
                cexpr, call = self.constraint_repr(constraint)
                if call:
                    self.add_line('if {}{}(subst{}):'.format(cvars, cexpr, self._substs))
                else:
                    self.add_line('if {}{}:'.format(cvars, cexpr))
                self.indent()
                self.add_line('pass')
                self._patterns = checked_patterns
                self.generate_constraints(remaining, checked_transitions)
                self.dedent()
            if remaining_patterns and remaining_transitions:
                self._patterns = remaining_patterns
                self.generate_constraints(remaining, remaining_transitions)
            self._patterns = remaining_patterns | checked_patterns

    def enter_global_constraint(self, constraint):
        cexpr, call = self.constraint_repr(constraint)
        if call:
            self.add_line('if {}(subst{}):'.format(cexpr, self._substs))
        else:
            self.add_line('if {}:'.format(cexpr))
        self.indent()
        self.add_line('pass')

    def constraint_repr(self, constraint):
        if isinstance(constraint, CustomConstraint) and isinstance(constraint.constraint, type(lambda: 0)):
            src = get_short_lambda_source(constraint.constraint)
            if src is not None:
                mapping = {k: v for v, k in constraint._variables.items()}
                params = constraint._variables.keys()
                pstr = r'\b({})\b'.format('|'.join(map(re.escape, params)))
                new_src = re.sub(pstr, lambda m: 'subst{}[{!r}]'.format(self._substs, constraint._variables[m[0]]), src)
                return new_src, False
        return repr(constraint), True

    def exit_global_constraint(self, constraint_index):
        self.dedent()

    def clean_code(self, code):
        return re.sub(r'\n(\s+)pass((?:\n\1#[^\n]*)*\n\1+\w)', r'\2', code)

    @staticmethod
    def _collapse_ifs(code):
        def sub_cb(m):
            indent = m['indent1']
            indent2 = indent + m['indent2']
            indent3 = indent2 + m['indent3']
            offset = len(indent3)
            inner = ('\n' + indent2).join(line[offset:] for line in m['block'].rstrip().split('\n'))
            result = '\n{}if {} and {}:\n{}{}\n'.format(indent, m['cond1'], m['cond2'], indent2, inner)
            if m['comment']:
                result = '\n{}{}{}'.format(indent, m['comment'].strip(), result)
            return result

        count = 1
        while count > 0:
            code, count = COLLAPSE_IF_RE.subn(sub_cb, code)
        return code
