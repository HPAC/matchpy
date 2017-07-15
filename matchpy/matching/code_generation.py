from ..expressions.expressions import Wildcard, AssociativeOperation
from ..expressions.functions import op_iter
from .syntactic import OPERATION_END, is_operation


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
        self.add_line('def match_root(subject):')
        self.indent()
        self.add_line('subjects{} = [subject]'.format(self._subjects))
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
                    self.add_line(
                        'yield {}, Substitution(subst{})'.format(self.final_label(pattern_index), self._substs)
                    )
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
        elif isinstance(transition.label, type):
            enter_func = self.enter_symbol_wildcard
            exit_func = self.exit_symbol_wildcard
        elif isinstance(transition.label, Wildcard):
            wc = transition.label
            if wc.fixed_size and self._associative_stack[-1] is None:
                enter_func = self.enter_fixed_wildcard
                exit_func = self.generate_state_code
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
        if transition.variable_name is not None:
            self.enter_variable_assignment(transition.variable_name, value)
        self.generate_state_code(transition.target)
        if transition.variable_name is not None:
            self.exit_variable_assignment()
        exit_func(value)
        self._patterns.update(removed)

    def get_args(self, operation):
        return 'list(op_iter({}))'.format(operation)

    def push_subjects(self, index=0):
        self.add_line(
            'subjects{} = {}'.format(self._subjects + 1, self.get_args('subjects{}[{}]'.format(self._subjects, index)))
        )
        self._subjects += 1

    def push_subst(self):
        new_subst = self.get_var_name('subst')
        self.add_line('subst{} = Substitution(subst{})'.format(self._substs + 1, self._substs))
        self._substs += 1

    def enter_operation(self, operation):
        self.add_line(
            'if len(subjects{0}) >= 1 and isinstance(subjects{0}[0], {1}):'.
            format(self._subjects, self.operation_symbol(operation))
        )
        self.indent()
        atype = operation if issubclass(operation, AssociativeOperation) else None
        self._associative_stack.append(atype)
        if atype is not None:
            self._associative += 1
            self.add_line('associative{} = type(subjects{}[0])'.format(self._associative, self._subjects))
        self.push_subjects()

    def operation_symbol(self, operation):
        return operation.__name__

    def exit_operation(self, value):
        self._subjects -= 1
        atype = self._associative_stack.pop()
        if atype is not None:
            self._associative -= 1
        self.dedent()

    def enter_symbol_wildcard(self, symbol_type):
        self.add_line(
            'if len(subjects{0}) >= 1 and isinstance(subjects{0}[0], {1}):'.
            format(self._subjects, self.symbol_type(symbol_type))
        )
        self.indent()
        tmp = self.get_var_name('tmp')
        self.add_line('{} = subjects{}.pop(0)'.format(tmp, self._subjects))
        return tmp

    def symbol_type(self, symbol):
        return symbol.__name__

    def exit_symbol_wildcard(self, value):
        self.add_line('subjects{}.insert(0, {})'.format(self._subjects, value))
        self.dedent()

    def enter_fixed_wildcard(self, wildcard):
        self.add_line('if len(subjects{}) >= 1:'.format(self._subjects))
        self.indent()
        tmp = self.get_var_name('tmp')
        self.add_line('{} = subjects{}.pop(0)'.format(tmp, self._subjects))
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
        self.add_line('{} = subjects{}.pop(0)'.format(tmp, self._subjects))
        return tmp

    def symbol_repr(self, symbol):
        return repr(symbol)

    def exit_symbol(self, value):
        self.add_line('subjects{}.insert(0, {})'.format(self._subjects, value))
        self.dedent()

    def enter_operation_end(self, _):
        self.add_line('if len(subjects{0}) == 0:'.format(self._subjects))
        self.indent()

    def exit_operation_end(self, value):
        self.dedent()

    def enter_sequence_wildcard(self, wildcard):
        i = self.get_var_name('i')
        self.add_line('for {} in range({}, len(subjects{}) + 1):'.format(i, wildcard.min_count, self._subjects))
        self.indent()
        tmp = self.get_var_name('tmp')
        self.add_line('{} = subjects{}[:{}]'.format(tmp, self._subjects, i))
        if self._associative_stack[-1] is not None:
            self.add_line('if len({}) > {}:'.format(tmp, wildcard.min_count))
            self.indent()
            self.add_line('{} = {}'.format(tmp, self.create_operation('associative{}'.format(self._associative), tmp)))
            self.dedent()
            self.add_line('elif len({}) == 1:'.format(tmp))
            self.indent()
            self.add_line('{0} = {0}[0]'.format(tmp))
            self.dedent()
        self.add_line('subjects{} = subjects{}[{}:]'.format(self._subjects + 1, self._subjects, i))
        self._subjects += 1
        return tmp

    def create_operation(self, operation, args):
        return 'create_operation_expression({}, {})'.format(operation, args)

    def exit_sequence_wildcard(self, value):
        self._subjects -= 1
        self.dedent()
