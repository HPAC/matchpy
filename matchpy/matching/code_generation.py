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
                    self.add_line('yield {}, Substitution(subst{})'.format(
                        self.final_label(pattern_index), self._substs
                    ))
            else:
                for transitions in state.transitions.values():
                    for transition in transitions:
                        self.generate_transition_code(transition)

    def generate_transition_code(self, transition):
        removed = self._patterns - transition.patterns
        self._patterns.intersection_update(transition.patterns)
        if is_operation(transition.label):
            state = self.enter_operation(transition.label)
            self.generate_state_code(transition.target)
            self.exit_operation(state)
        elif isinstance(transition.label, type):
            state = self.enter_symbol_wildcard(transition.label)
            self.generate_state_code(transition.target)
            self.exit_symbol_wildcard(state)
        elif isinstance(transition.label, Wildcard):
            wc = transition.label
            if wc.fixed_size and self._associative_stack[-1] is None:
                state = self.enter_fixed_wildcard(wc, transition.variable_name)
                self.generate_state_code(transition.target)
                self.exit_fixed_wildcard(state)
            else:
                state = self.enter_sequence_wildcard(wc, transition.variable_name)
                self.generate_state_code(transition.target)
                self.exit_sequence_wildcard(state)
        elif transition.label is OPERATION_END:
            state = self.enter_operation_end()
            self.generate_state_code(transition.target)
            self.exit_operation_end(state)
        else:
            state = self.enter_symbol(transition.label, transition.variable_name)
            self.generate_state_code(transition.target)
            self.exit_symbol(state)
        self._patterns.update(removed)

    def get_args(self, operation):
        return 'list(op_iter({}))'.format(operation)

    def push_subjects(self, index=0):
        self.add_line('subjects{} = {}'.format(
            self._subjects + 1, self.get_args('subjects{}[{}]'.format(self._subjects, index))
        ))
        self._subjects += 1

    def push_subst(self):
        new_subst = self.get_var_name('subst')
        self.add_line('subst{} = Substitution(subst{})'.format(
            self._substs + 1, self._substs
        ))
        self._substs += 1

    def enter_operation(self, operation):
        self.add_line('if len(subjects{0}) >= 1 and isinstance(subjects{0}[0], {1}):'.format(
            self._subjects, self.operation_symbol(operation)
        ))
        self.indent()
        atype = operation if issubclass(operation, AssociativeOperation) else None
        self._associative_stack.append(atype)
        if atype is not None:
            self._associative += 1
            self.add_line('associative{} = type(subjects{}[0])'.format(self._associative, self._subjects))
        self.push_subjects()

    def operation_symbol(self, operation):
        return operation.__name__

    def exit_operation(self, state):
        self._subjects -= 1
        atype = self._associative_stack.pop()
        if atype is not None:
            self._associative -= 1
        self.dedent()

    def enter_symbol_wildcard(self, symbol_type):
        self.add_line('if len(subjects{0}) >= 1 and isinstance(subjects{0}[0], {1}):'.format(
            self._subjects, self.symbol_type(symbol_type)
        ))
        self.indent()
        tmp = self.get_var_name('tmp')
        self.add_line('{} = subjects{}.pop(0)'.format(tmp, self._subjects))
        return tmp

    def symbol_type(self, symbol):
        return symbol.__name__

    def exit_symbol_wildcard(self, state):
        self.add_line('subjects{}.insert(0, {})'.format(self._subjects, state))
        self.dedent()

    def enter_fixed_wildcard(self, wildcard, variable_name):
        self.add_line('if len(subjects{}) >= 1:'.format(self._subjects))
        self.indent()
        tmp = self.get_var_name('tmp')
        self.add_line('{} = subjects{}.pop(0)'.format(tmp, self._subjects))
        if variable_name is not None:
            self.push_subst()
            self.add_line('subst{}.try_add_variable({!r}, {})'.format(
                self._substs, variable_name, tmp))
        return (tmp, variable_name)

    def exit_fixed_wildcard(self, state):
        tmp, vname = state
        if vname is not None:
            self._substs -= 1
        self.add_line('subjects{}.insert(0, {})'.format(self._subjects, tmp))
        self.dedent()

    def enter_symbol(self, symbol, variable_name):
        self.add_line('if len(subjects{0}) >= 1 and subjects{0}[0] == {1}:'.format(
            self._subjects, self.symbol_repr(symbol)))
        self.indent()
        tmp = self.get_var_name('tmp')
        self.add_line('{} = subjects{}.pop(0)'.format(tmp, self._subjects))
        if variable_name is not None:
            self.push_subst()
            self.add_line('subst{}.try_add_variable({!r}, {})'.format(
                self._substs, variable_name, tmp))
        return (tmp, variable_name)

    def symbol_repr(self, symbol):
        return repr(symbol)

    def exit_symbol(self, state):
        tmp, vname = state
        if vname is not None:
            self._substs -= 1
        self.add_line('subjects{}.insert(0, {})'.format(self._subjects, tmp))
        self.dedent()

    def enter_operation_end(self):
        self.add_line('if len(subjects{0}) == 0:'.format(self._subjects))
        self.indent()

    def exit_operation_end(self, state):
        self.dedent()

    def enter_sequence_wildcard(self, wildcard, variable_name):
        i = self.get_var_name('i')
        self.add_line('for {} in range({}, len(subjects{}) + 1):'.format(
            i, wildcard.min_count, self._subjects))
        self.indent()
        tmp = self.get_var_name('tmp')
        self.add_line('{} = subjects{}[:{}]'.format(tmp, self._subjects, i))
        if self._associative_stack[-1] is not None:
            self.add_line('if len({}) > {}:'.format(tmp, wildcard.min_count))
            self.indent()
            self.add_line('{} = {}'.format(
                tmp, self.create_operation('associative{}'.format(self._associative), tmp)))
            self.dedent()
            self.add_line('elif len({}) == 1:'.format(tmp))
            self.indent()
            self.add_line('{0} = {0}[0]'.format(tmp))
            self.dedent()
        self.add_line('subjects{} = subjects{}[{}:]'.format(
            self._subjects + 1, self._subjects, i))
        self._subjects += 1
        if variable_name is not None:
            self.push_subst()
            self.add_line('subst{}.try_add_variable({!r}, {})'.format(
                self._substs, variable_name, tmp))
        return variable_name

    def create_operation(self, operation, args):
        return 'create_operation_expression({}, {})'.format(operation, args)

    def exit_sequence_wildcard(self, state):
        if state is not None:
            self._substs -= 1
        self._subjects -= 1
        self.dedent()

