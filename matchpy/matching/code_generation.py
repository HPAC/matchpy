from ..expressions.expressions import Wildcard
from .syntactic import OPERATION_END, is_operation


class CodeIndentor:
    def __init__(self, indentation='\t'):
        self.indentation = indentation
        self.level = 0
        self.code = ''

    def indent(self):
        self.level += 1

    def dedent(self):
        if self.level > 0:
            self.level -= 1

    def add_line(self, line):
        self.code += (self.indentation * self.level) + str(line) + '\n'

    def add(self, code):
        for line in str(code).rstrip().split('\n'):
            self.code += (self.indentation * self.level) + line + '\n'

    def __str__(self):
        return self.code


_var_number = 0

def get_var_name(prefix):
    global _var_number
    _var_number += 1
    return prefix + str(_var_number)


def generate_code(matcher):
    code = CodeIndentor()
    code.add_line('_PATTERNS = [')
    code.indent()
    for pattern in matcher.patterns:
        code.add_line(repr(pattern) + ',')
    code.dedent()
    code.add_line(']')
    code.add_line('def match_root(subject):')
    code.indent()
    subjects = get_var_name('subjects')
    subst = get_var_name('subst')
    code.add_line('{} = [subject]'.format(subjects))
    code.add_line('{} = Substitution()'.format(subst))
    code.add(generate_state_code(matcher, matcher.root, subjects, subst, set(range(len(matcher.patterns)))))
    code.dedent()
    return code


def generate_state_code(matcher, state, subjects, subst, patterns):
    code = CodeIndentor()
    if state.matcher is not None:
        raise NotImplementedError()
    else:
        code.add_line('# State {}'.format(state.number))
        if state.number in matcher.finals:
            for pattern_index in patterns:
                code.add_line('yield _PATTERNS[{}][1], Substitution({})'.format(pattern_index, subst))
        else:
            for transitions in state.transitions.values():
                for transition in transitions:
                    code.add(generate_transition_code(matcher, transition, subjects, subst, patterns))
    return code



def generate_transition_code(matcher, transition, subjects, subst, patterns):
    code = CodeIndentor()
    new_patterns = transition.patterns.intersection(patterns)
    if is_operation(transition.label):
        name = transition.label.__name__
        code.add_line('if len({0}) >= 1 and isinstance({0}[0], {1}):'.format(subjects, name))
        code.indent()
        new_subjects = get_var_name('subjects')
        code.add_line('{} = list({}[0].operands)'.format(new_subjects, subjects))
        code.add(generate_state_code(matcher, transition.target, new_subjects, subst, new_patterns))
        code.dedent()
    elif isinstance(transition.label, type):
        name = transition.label.__name__
        code.add_line('if len({0}) >= 1 and isinstance({0}[0], {1}):'.format(subjects, name))
        code.indent()
        tmp = get_var_name('tmp')
        code.add_line('{} = {}.pop(0)'.format(tmp, subjects))
        code.add(generate_state_code(matcher, transition.target, subjects, subst, new_patterns))
        code.add_line('{}.insert(0, {})'.format(subjects, tmp))
        code.dedent()
    elif isinstance(transition.label, Wildcard):
        wc = transition.label
        if wc.fixed_size:
            code.add_line('if len({}) >= 1:'.format(subjects))
            code.indent()
            tmp = get_var_name('tmp')
            code.add_line('{} = {}.pop(0)'.format(tmp, subjects))
            new_subst = subst
            if transition.variable_name is not None:
                new_subst = get_var_name('subst')
                code.add_line('{} = Substitution({})'.format(new_subst, subst))
                code.add_line('{}.try_add_variable({!r}, {})'.format(new_subst, transition.variable_name, tmp))
            code.add(generate_state_code(matcher, transition.target, subjects, new_subst, new_patterns))
            code.add_line('{}.insert(0, {})'.format(subjects, tmp))
            code.dedent()
        else:
            raise NotImplementedError()
    elif transition.label is OPERATION_END:
        code.add_line('if len({0}) == 0:'.format(subjects))
        code.indent()
        code.add(generate_state_code(matcher, transition.target, subjects, subst, new_patterns))
        code.dedent()
    else:
        code.add_line('if len({0}) >= 1 and {0}[0] == {1!r}:'.format(subjects, transition.label))
        code.indent()
        tmp = get_var_name('tmp')
        code.add_line('{} = {}.pop(0)'.format(tmp, subjects))
        code.add(generate_state_code(matcher, transition.target, subjects, subst, new_patterns))
        code.add_line('{}.insert(0, {})'.format(subjects, tmp))
        code.dedent()
    return code
