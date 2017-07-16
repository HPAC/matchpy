from ..expressions.expressions import Wildcard, AssociativeOperation, SymbolWildcard
from ..expressions.functions import op_iter
from .syntactic import OPERATION_END, is_operation
from .many_to_one import _EPS


class CodeGenerator:
    def __init__(self, matcher):
        self._matcher = matcher
        self._var_number = 0
        self._indentation = '\t'
        self._level = 0
        self._code = ''
        self._subjects = 0
        self._substs = 0
        self._patterns = set(range(len(matcher.patterns)))
        self._associative = 0
        self._associative_stack = [None]

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

    def generate_code(self):
        self.add_line('from collections import deque')
        self.add_line('def match_root(subject):')
        self.indent()
        self.add_line('subjects{} = deque([subject])'.format(self._subjects))
        self.add_line('subst{} = Substitution()'.format(self._substs))
        self.generate_state_code(self._matcher.root)
        self.dedent()
        return self._code

    def final_label(self, index):
        return str(index)

    def generate_state_code(self, state):
        if state.matcher is not None:
            raise NotImplementedError()
        else:
            self.add_line('# State {}'.format(state.number))
            if state.number in self._matcher.finals:
                for pattern_index in self._patterns:
                    self.add_line('if len(subjects{}) == 0:'.format(self._subjects))
                    self.indent()
                    self.yield_final_substitution(pattern_index)
                    self.dedent()
            else:
                for transitions in state.transitions.values():
                    for transition in transitions:
                        self.generate_transition_code(transition)

    def generate_transition_code(self, transition):
        removed = self._patterns - transition.patterns
        self._patterns.intersection_update(transition.patterns)
        enter_func = None
        exit_func = None
        if is_operation(transition.label):
            enter_func = self.enter_operation
            exit_func = self.exit_operation
        # elif isinstance(transition.label, type):
        #     enter_func = self.enter_symbol_wildcard
        #     exit_func = self.exit_symbol_wildcard
        elif transition.label == _EPS:
            enter_func = self.enter_eps
            exit_func = self.exit_eps
        elif isinstance(transition.label, Wildcard):
            wc = transition.label
            if wc.optional is not None:
                self.enter_variable_assignment(transition.variable_name, self.optional_expr(wc.optional))
                self.generate_state_code(transition.target)
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
            # var_value = 'tuple({})'.format(value) if isinstance(transition.label, Wildcard) and not transition.label.fixed_size else value
            self.enter_variable_assignment(transition.variable_name, var_value)
        if transition.subst is not None:
            self.enter_subst(transition.subst)
        self.generate_state_code(transition.target)
        if transition.subst is not None:
            self.exit_subst(transition.subst)
        if transition.variable_name is not None:
            self.exit_variable_assignment()
        exit_func(value)
        self._patterns.update(removed)

    def get_args(self, operation):
        return 'deque(op_iter({}))'.format(operation)

    def push_subjects(self, value):
        self.add_line('subjects{} = {}'.format(self._subjects + 1, self.get_args(value)))
        self._subjects += 1

    def push_subst(self):
        new_subst = self.get_var_name('subst')
        self.add_line('subst{} = Substitution(subst{})'.format(self._substs + 1, self._substs))
        self._substs += 1

    def enter_eps(self, _):
        return 'subjects{0}'.format(self._subjects)

    def exit_eps(self, _):
        pass

    def enter_operation(self, operation):
        self.add_line(
            'if len(subjects{0}) >= 1 and isinstance(subjects{0}[0], {1}):'.
            format(self._subjects, self.operation_symbol(operation))
        )
        self.indent()
        tmp = self.get_var_name('tmp')
        self.add_line('{} = subjects{}.popleft()'.format(tmp, self._subjects))
        atype = operation if issubclass(operation, AssociativeOperation) else None
        self._associative_stack.append(atype)
        if atype is not None:
            self._associative += 1
            self.add_line('associative{} = {}'.format(self._associative, tmp))
            self.add_line('associative_type{} = type({})'.format(self._associative, tmp))
        self.push_subjects(tmp)
        return tmp

    def operation_symbol(self, operation):
        return operation.__name__

    def exit_operation(self, value):
        self._subjects -= 1
        self.add_line('subjects{}.insert(0, {})'.format(self._subjects, value))
        self.dedent()
        atype = self._associative_stack.pop()
        if atype is not None:
            self._associative -= 1

    def enter_symbol_wildcard(self, wildcard):
        self.add_line(
            'if len(subjects{0}) >= 1 and isinstance(subjects{0}[0], {1}):'.
            format(self._subjects, self.symbol_type(wildcard.symbol_type))
        )
        self.indent()
        tmp = self.get_var_name('tmp')
        self.add_line('{} = subjects{}.popleft()'.format(tmp, self._subjects))
        return tmp

    def symbol_type(self, symbol):
        return symbol.__name__

    def exit_symbol_wildcard(self, value):
        self.add_line('subjects{}.insert(0, {})'.format(self._subjects, value))
        self.dedent()

    def enter_fixed_wildcard(self, wildcard):
        print(wildcard)
        self.add_line('if len(subjects{}) >= 1:'.format(self._subjects))
        self.indent()
        tmp = self.get_var_name('tmp')
        self.add_line('{} = subjects{}.popleft()'.format(tmp, self._subjects))
        return tmp

    def exit_fixed_wildcard(self, value):
        self.add_line('subjects{}.insert(0, {})'.format(self._subjects, value))
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
        self.add_line(
            'if len(subjects{0}) >= 1 and subjects{0}[0] == {1}:'.format(self._subjects, self.symbol_repr(symbol))
        )
        self.indent()
        tmp = self.get_var_name('tmp')
        self.add_line('{} = subjects{}.popleft()'.format(tmp, self._subjects))
        return tmp

    def symbol_repr(self, symbol):
        return repr(symbol)

    def exit_symbol(self, value):
        self.add_line('subjects{}.insert(0, {})'.format(self._subjects, value))
        self.dedent()

    def enter_operation_end(self, _):
        self.add_line('if len(subjects{0}) == 0:'.format(self._subjects))
        self.indent()
        self._subjects -= 1
        atype = self._associative_stack.pop()
        if atype is not None:
            self._associative -= 1
        return atype

    def exit_operation_end(self, atype):
        self._subjects += 1
        self.dedent()
        self._associative_stack.append(atype)
        if atype is not None:
            self._associative += 1

    def enter_sequence_wildcard(self, wildcard):
        # i = self.get_var_name('i')
        tmp = self.get_var_name('tmp')
        tmp2 = self.get_var_name('tmp')
        mc = wildcard.min_count if wildcard.optional is None or wildcard.min_count > 0 else 1
        self.add_line('if len(subjects{}) >= {}:'.format(self._subjects, mc))
        self.indent()
        self.add_line('{} = []'.format(tmp))
        for _ in range(mc):
            self.add_line('{}.append(subjects{}.popleft())'.format(tmp, self._subjects))
        self.add_line('while True:')
        self.indent()
        # self.add_line('{} = tuple(subjects{}[:{}])'.format(tmp, self._subjects, i))
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
            self.add_line('raise NotImplementedError()')
            self.dedent()
        else:
            self.add_line('{} = tuple({})'.format(tmp2, tmp))

        # self.add_line('subjects{} = subjects{}[{}:]'.format(self._subjects + 1, self._subjects, i))
        # self._subjects += 1
        return tmp, tmp2

    def create_operation(self, operation, operation_type, args):
        return 'create_operation_expression({}, {})'.format(operation, args)

    def exit_sequence_wildcard(self, value):
        # self._subjects -= 1
        self.add_line('if len(subjects{}) == 0:'.format(self._subjects))
        self.indent()
        self.add_line('break')
        self.dedent()
        self.add_line('{}.append(subjects{}.popleft())'.format(value, self._subjects))
        self.dedent()
        self.add_line('subjects{}.extendleft(reversed({}))'.format(self._subjects, value))
        self.dedent()

    def yield_final_substitution(self, pattern_index):
        renaming = self._matcher.pattern_vars[pattern_index]
        self.add_line('tmp_subst = Substitution()')
        for original, renamed in renaming.items():
            self.add_line('tmp_subst[{!r}] = subst{}[{!r}]'.format(original, self._substs, renamed))
        self.add_line('# {}'.format(self._matcher.patterns[pattern_index][0]))
        self.add_line('yield {}, tmp_subst'.format(self.final_label(pattern_index)))
