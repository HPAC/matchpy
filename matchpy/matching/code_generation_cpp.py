import re
from collections import defaultdict

from matchpy.expressions.expressions import Wildcard, AssociativeOperation, SymbolWildcard
from matchpy.expressions.constraints import CustomConstraint
from matchpy.expressions.functions import op_iter, get_variables
from matchpy.matching.syntactic import OPERATION_END, is_operation
from matchpy.matching.many_to_one import _EPS
from matchpy.utils import get_short_lambda_source

COLLAPSE_IF_RE = re.compile(
    r'\n(?P<indent1>\s*)if (?P<cond1>[^\n]+):\n+\1(?P<indent2>\s+)'
    r'(?P<comment>(?:\#[^\n]*\n+\1\3)*)'
    r'if (?P<cond2>[^\n]+):\n+'
    r'(?P<block>\1\3(?P<indent3>\s+)[^\n]*\n+(?:\1\3\7[^\n]*\n+)*)'
    r'(?!\1(?:\3|elif|else))'
)


class _IndentedCodePrinter:
    # Abstract class
    def __init__(self, level):
        self._level = level
        self._code = ""
        self._indentation = '    '

    def indent(self, bracket=True):
        if bracket:
            self.add_line("{")
        self._level += 1

    def dedent(self, bracket=True):
        self._level -= 1
        if bracket:
            self.add_line("}")

    def add_line(self, line):
        code = (self._indentation * self._level) + str(line) + '\n'
        self._code += code


class StatePartMethod(_IndentedCodePrinter):
    state_name_counter = defaultdict(int)

    def __init__(self, state_name):
        super(StatePartMethod, self).__init__(level=1)
        self.state_name = state_name
        self.state_name_counter[state_name] += 1
        part = self.state_name_counter[state_name] - 1
        self.part = part
        self.add_line("void {}()".format(self._get_state_print_form(state_name, part)))
        self.indent()

    def generate_code(self):
        dest = 1
        while self._level > dest:
            self.dedent()
        return self._code

    def add_pointer(self, other_code):
        state_name = other_code.state_name
        part = other_code.part
        pointer_template = "current = std::bind(&match_root::{}, this);"
        self.add_line(pointer_template.format(
            self._get_state_print_form(state_name, part)
        ))
        #if self._level > 2:
            #self.add_line('return;')

    def _get_state_print_form(self, name, part):
        if part > 0:
            return "{0}part{1:03}".format(name, part)
        else:
            return "{0}".format(name)

    def add_transition(self, transition_target_code, return_target_code):
        self.add_pointer(transition_target_code)
        self.dedent()
        self.add_line("// add_transition::self.add_pointer(return_target_code)")
        self.add_pointer(return_target_code)


class CppCodeGenerator(_IndentedCodePrinter):
    def __init__(self, matcher):
        super(CppCodeGenerator, self).__init__(level=0)
        self._matcher = matcher
        self._var_number = 0
        self._indentation = '    '
        self._code = ''
        self._subjects = ['subjects']
        self._substs = 0
        self._patterns = set(range(len(matcher.patterns)))
        self._associative = 0
        self._associative_stack = [None]
        self._global_code = []
        self._imports = set()
        self._class_fields = {}
        self._method_list = []

    def add_class_field(self, vartype, name):
        if name in self._class_fields:
            assert self._class_fields[name] == vartype
            return
        self._class_fields[name] = vartype

    def add_new_state(self, name):
        if not isinstance(name, str):
            name = "state{}".format(name.number)
        state_printer = StatePartMethod(name)
        self._method_list.append(state_printer)
        return state_printer

    def add_new_state_part(self, state_code):
        #StatePartMethod.state_name_counter[state_code.state_name] += 1
        #part = StatePartMethod.state_name_counter[state_code.state_name]
        state_part = StatePartMethod(state_code.state_name)
        self._method_list.append(state_part)
        return state_part

    def add_new_state_return(self, state_code, target_state):
        state_part = self.add_new_state_part(state_code)
        state_part.add_line("// Return from state {0.target.number}".format(target_state))
        return state_part

    def get_var_name(self, prefix):
        self._var_number += 1
        return prefix + str(self._var_number)

    def generate_code(self, func_name='match_root', add_imports=True):
        self._imports.add('#include <deque>')
        self._imports.add('#include <iostream>')
        self._imports.add('#include <tuple>')
        self._imports.add('#include <map>')
        self._imports.add('#include <string>')
        self._imports.add('#include <deque>')
        self._imports.add('#include <functional>')

        self._imports.add('#include <symengine/basic.h>')
        self._imports.add('#include <symengine/pow.h>')

        self._imports.add('#include "generator_trick.h"')

        self.add_line('')
        self.add_line('using namespace std;')
        self.add_line('using namespace SymEngine;')
        self.add_line('typedef map<string, RCP<const Basic>> Substitution;')
        self.add_line('typedef deque<RCP<const Basic>> Deque;')
        self.add_line('')

        self._code += """
int try_add_variable(Substitution &subst, string variable_name,
                     RCP<const Basic> &replacement)
{
    if (subst.find(variable_name) == subst.end()) {
        subst[variable_name] = replacement;
    } else {
    }
    return 0;
}

Deque get_deque(RCP<const Basic> expr)
{
    Deque d;
    for (RCP<const Basic> i : expr->get_args()) {
        d.push_back(i);
    }
    return d;
}

RCP<const Basic> x = symbol("x");
RCP<const Basic> y = symbol("y");
RCP<const Basic> z = symbol("z");
RCP<const Basic> w = symbol("w");

"""
        self.add_line('class {} : public GeneratorTrick<tuple<int, Substitution>>'.format(func_name))
        self.add_line('{')
        self.add_line('public:')
        self.indent(bracket=False)
        self.add_line('{}(RCP<const Basic> &subject)'.format(func_name))
        self.indent()
        self.add_line('subjects.push_back(subject);')
        self.dedent()
        self.add_line('virtual ~{}(){{}};'.format(func_name))
        self.dedent(bracket=False)
        self.add_line("")
        self.add_line('private:')
        self.indent(bracket=False)
        self.add_class_field("Deque", "subjects")

        start = self.add_new_state("start")
        stop = self.add_new_state("stop")
        first_state = self.add_new_state(self._matcher.root)

        start.add_pointer(first_state)
        stop.add_line("generator_stop = true;")

        last_method = self.generate_state_code(self._matcher.root, first_state)
        last_method.add_pointer(stop)

        self.generate_state_methods()
        self.generate_class_fields()
        self.dedent(False)
        self.add_line('};')

        if add_imports:
            self._global_code.insert(0, '\n'.join(self._imports))

        return self.clean_code('\n\n'.join(p for p in self._global_code if p)), self.clean_code(self._code)

    def generate_state_methods(self):
        for state_code in self._method_list:
            self._code += state_code.generate_code()

    def generate_class_fields(self):
        for name, vartype in self._class_fields.items():
            self.add_line('{0} {1};'.format(vartype, name))

    def final_label(self, index, subst_name):
        return str(index)

    def generate_state_code(self, state, state_code):
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
class CommutativeMatcher{0} : public CommutativeMatcher {{
{8}CommutativeMatcher{0} _instance;
{8}patterns = {1};
{8}Deque subjects = {2};
{8}Deque subjects_by_id = {7};
{8}BipartiteGraph bipartite;
{8}bool associative = {3};
{8}int max_optional_count = {4};
{8}anonymous_patterns = {5};

{8}CommutativeMatcher{0}() {{
{8}{8}// self.add_subject(None)
{8}}}

{8}@staticmethod
{8}CommutativeMatcher{0} get() {{
{8}{8}return _instance;
{8}}}

{8}static {6}'''.strip().format(
                    state.number, patterns, subjects, associative, max_optional_count, anonymous_patterns, code,
                    subjects_by_id, self._indentation
                )
            )
            state_code.add_line('matcher = CommutativeMatcher{}.get()'.format(state.number))
            tmp = self.get_var_name('tmp')
            self.add_class_field('RCP<const Basic>', tmp)
            state_code.add_line('{} = {}'.format(tmp, self._subjects[-1]))
            state_code.add_line('{} = []'.format(self._subjects[-1]))
            state_code.add_line('for s in {}:'.format(tmp))
            state_code.indent()
            state_code.add_line('matcher.add_subject(s)')
            subjects = self._subjects.pop()
            state_code.dedent()
            state_code.add_line(
                'for pattern_index, subst{} in matcher.match({}, subst{}):'.format(self._substs + 1, tmp, self._substs)
            )
            self._substs += 1
            state_code.indent()
            state_code.add_line('pass')
            for pattern_index, transitions in state.transitions.items():
                state_code.add_line('if pattern_index == {}:'.format(pattern_index))
                state_code.indent()
                state_code.add_line('pass')
                patterns, variables = next((p, v) for i, p, v in state.matcher.patterns.values() if i == pattern_index)
                variables = set(v[0][0] for v in variables)
                pvars = iter(get_variables(state.matcher.automaton.patterns[i][0].expression) for i in patterns)
                variables.update(*pvars)
                constraints = []
                if variables:
                    constraints = sorted(
                        set.union(*iter(self._matcher.constraint_vars.get(v, set()) for v in variables))
                    )
                self.generate_constraints(state_code, constraints, transitions)
                state_code.dedent()
            state_code.dedent()
            self._substs -= 1
            self._subjects.append(subjects)
        else:
            state_code.add_line('// State {}'.format(state.number))
            state_code.add_line('cout << "State {}" << endl;'.format(state.number))
            if state.number in self._matcher.finals:
                state_code.add_line('if ({}.size() == 0) {{'.format(self._subjects[-1]))
                state_code.indent(bracket=False)
                for pattern_index in self._patterns:
                    constraints = self._matcher.patterns[pattern_index][0].global_constraints
                    for constraint in constraints:
                        self.enter_global_constraint(constraint)
                    self.yield_final_substitution(state_code, pattern_index)
                    for constraint in constraints:
                        self.exit_global_constraint(constraint)
                state_code.dedent()
            else:
                for transitions in state.transitions.values():
                    for transition in transitions:
                        #next_method = self.add_new_state_part(state_code)
                        #next_method.add_line("// generate_state_code::next_method 289")
                        next_method = self.generate_transition_code(state_code, transition)
                        state_code = next_method

        next_method = self.add_new_state_part(state_code)
        state_code.add_pointer(next_method)
        return next_method

    def commutative_var_entry(self, entry):
        return '(VariableWithCount("{}", {}, {}, {}), {})'.format(
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

    def generate_transition_code(self, state_code, transition):
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
                raise ValueError
                self.enter_variable_assignment(state_code, transition.variable_name, self.optional_expr(wc.optional))
                constraints = sorted(transition.check_constraints) if transition.check_constraints is not None else []
                next_method = self.generate_constraints(state_code, constraints, [transition])
                next_method.add_pointer(state_return)
                self.exit_variable_assignment(state_return, next_method)
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

        value = enter_func(state_code, transition.label)
        value, var_value = value if isinstance(value, tuple) else (value, value)
        if transition.variable_name is not None:
            self.enter_variable_assignment(state_code, transition.variable_name, var_value)
        if transition.subst is not None:
            self.enter_subst(state_code, transition.subst)
        constraints = sorted(transition.check_constraints) if transition.check_constraints is not None else []
        ret_method = self.generate_constraints(state_code, constraints, [transition])

        transition_return = self.add_new_state_part(state_code)
        next_method = self.add_new_state_part(state_code)
        ret_method.add_pointer(transition_return)
        transition_return.add_line("// line 312")

        state_code.add_line('} else {')
        state_code.indent(bracket=False)
        state_code.add_line("// 357")
        state_code.add_pointer(next_method)
        state_code.dedent()

        if transition.subst is not None:
            self.exit_subst(transition.subst)
        if transition.variable_name is not None:
            self.exit_variable_assignment(state_code, transition_return)
        exit_func(transition_return, value)

        #state_code.add_pointer(transition_return)
        transition_return.add_line("// generate_transition_code transition_return.add_pointer(next_method)")
        transition_return.add_pointer(next_method)
        return next_method

    def push_subjects(self, state_code, value, operation):
        self._subjects.append(self.get_var_name('subjects'))
        self.add_class_field('Deque', self._subjects[-1])
        #state_code.add_line('{}.clear();'.format(self._subjects[-1]))
        #state_code.add_line('{}.push_front({});'.format(self._subjects[-1], operation))
        state_code.add_line('{} = get_deque({});'.format(self._subjects[-1], value))
        #self.get_args(value, operation)))

    def push_subst(self, state_code):
        new_subst = self.get_var_name('subst')
        self.add_class_field("Substitution", new_subst)
        state_code.add_line('subst{} = Substitution(subst{});'.format(self._substs + 1, self._substs))
        self._substs += 1

    def enter_eps(self, state_code, _):
        return '{0}'.format(self._subjects[-1])

    def exit_eps(self, state_code, _):
        pass

    def enter_operation(self, state_code, operation):
        state_code.add_line(
            'if ({0}.size() >= 1 && is_a<{1}>(*{0}[0])) {{'.format(
                self._subjects[-1], self.operation_symbol(operation))
        )
        state_code.indent(bracket=False)
        tmp = self.get_var_name('tmp')
        self.add_class_field('RCP<const Basic>', tmp)
        state_code.add_line('{} = {}.front();'.format(tmp, self._subjects[-1]))
        state_code.add_line('{}.pop_front();'.format(self._subjects[-1]))
        atype = operation if issubclass(operation, AssociativeOperation) else None
        self._associative_stack.append(atype)
        if atype is not None:
            self._associative += 1
            state_code.add_line('associative{} = {}'.format(self._associative, tmp))
            state_code.add_line('associative_type{} = type({})'.format(self._associative, tmp))
        self.push_subjects(state_code, tmp, operation)
        return tmp

    def operation_symbol(self, operation):
        if operation is None:
            return 'None'
        return operation.__name__

    def exit_operation(self, return_code, value):
        self._subjects.pop()
        return_code.add_line('{}.push_front({});'.format(self._subjects[-1], value))
        atype = self._associative_stack.pop()
        if atype is not None:
            self._associative -= 1

    def enter_symbol_wildcard(self, state_code, wildcard):
        state_code.add_line(
            'if ({0}.size() >= 1 && is_a<{1}(*{0}[0])) {{'.
            format(self._subjects[-1], self.symbol_type(wildcard.symbol_type))
        )
        state_code.indent(bracket=False)
        tmp = self.get_var_name('tmp')
        self.add_class_field('RCP<const Basic>', tmp)
        state_code.add_line('{} = {}.front();'.format(tmp, self._subjects[-1]))
        state_code.add_line('{}.pop_front();'.format(self._subjects[-1]))
        return tmp

    def symbol_type(self, symbol):
        return symbol.__name__

    def exit_symbol_wildcard(self, state_code, value):
        state_code.add_line('{}.push_front({});'.format(self._subjects[-1], value))

    def enter_fixed_wildcard(self, state_code, wildcard):
        state_code.add_line('if ({}.size() >= 1) {{'.format(self._subjects[-1]))
        state_code.indent(bracket=False)
        tmp = self.get_var_name('tmp')
        self.add_class_field('RCP<const Basic>', tmp)
        state_code.add_line('{} = {}.front();'.format(tmp, self._subjects[-1]))
        state_code.add_line('{}.pop_front();'.format(self._subjects[-1]))
        return tmp

    def exit_fixed_wildcard(self, state_code, value):
        state_code.add_line('{}.push_front({});'.format(self._subjects[-1], value))

    def enter_variable_assignment(self, state_code, variable_name, value):
        self.push_subst(state_code)
        state_code.add_line('if (!try_add_variable(subst{}, "{}", {})) {{'.format(self._substs, variable_name, value))
        state_code.indent(bracket=False)

    def enter_subst(self, state_code, subst):
        self.push_subst(state_code)
        state_code.add_line('try:')
        state_code.indent()
        for name, value in subst.items():
            state_code.add_line('try_add_variable(subst{}, "{}", {});'.format(self._substs, name, self.expr(value)))
        state_code.dedent()
        state_code.add_line('except ValueError:')
        state_code.indent()
        state_code.add_line('pass')
        state_code.dedent()
        state_code.add_line('else:')
        state_code.indent()
        state_code.add_line('pass')

    def expr(self, expr):
        return repr(expr)

    def exit_subst(self, subst):
        self._substs -= 1

    def exit_variable_assignment(self, state_code, next_method):
        state_code.add_line("// exit_variable_assignment")
        state_code.add_pointer(next_method)
        self._substs -= 1

    def enter_optional_wildcard(self, wildcard, variable_name):
        self.enter_variable_assignment(variable_name, self.optional_expr(wildcard.optional))

    def optional_expr(self, expr):
        return repr(expr)

    def exit_optional_wildcard(self, state_code, value):
        self.exit_variable_assignment(state_code)

    def enter_symbol(self, state_code, symbol):
        state_code.add_line('if ({0}.size() >= 1 && {0}[0]->__eq__(*{1})) {{'.format(self._subjects[-1], self.symbol_repr(symbol)))
        state_code.indent(bracket=False)
        tmp = self.get_var_name('tmp')
        self.add_class_field('RCP<const Basic>', tmp)
        state_code.add_line('{} = {}.front();'.format(tmp, self._subjects[-1]))
        state_code.add_line('{}.pop_front();'.format(self._subjects[-1]))
        return tmp

    def symbol_repr(self, symbol):
        return repr(symbol)

    def exit_symbol(self, state_code, value):
        state_code.add_line('{}.push_front({});'.format(self._subjects[-1], value))

    def enter_operation_end(self, state_code, _):
        state_code.add_line('if ({0}.size() == 0) {{'.format(self._subjects[-1]))
        state_code.indent(bracket=False)
        subjects = self._subjects.pop()
        atype = self._associative_stack.pop()
        if atype is not None:
            self._associative -= 1
        return [subjects, atype]

    def exit_operation_end(self, state_code, value):
        subjects, atype = value
        self._subjects.append(subjects)
        self._associative_stack.append(atype)
        if atype is not None:
            self._associative += 1

    def enter_sequence_wildcard(self, state_code, wildcard):
        tmp = self.get_var_name('tmp')
        self.add_class_field('RCP<const Basic>', tmp)
        tmp2 = self.get_var_name('tmp')
        self.add_class_field('RCP<const Basic>', tmp)
        mc = wildcard.min_count if wildcard.optional is None or wildcard.min_count > 0 else 1
        state_code.add_line('if ({}.size() >= {}) {{'.format(self._subjects[-1], mc))
        state_code.indent(bracket=False)
        state_code.add_line('{} = []'.format(tmp))
        for _ in range(mc):
            state_code.add_line('{}.push_back({}.front());'.format(tmp, self._subjects[-1]))
            state_code.add_line('{}.pop_front();'.format(self._subjects[-1]))
        state_code.add_line('while True:')
        state_code.indent()
        if self._associative_stack[-1] is not None and wildcard.fixed_size:
            state_code.add_line('if ({}.size() > {}) {{'.format(tmp, wildcard.min_count))
            state_code.indent(bracket=False)
            state_code.add_line(
                '{} = {}'.format(
                    tmp2,
                    self.create_operation(
                        'associative{}'.format(self._associative), 'associative{}'.format(self._associative), tmp
                    )
                )
            )
            state_code.dedent()
            state_code.add_line('else if ({}.size() == 1) {{'.format(tmp))
            state_code.indent(bracket=False)
            state_code.add_line('{} = {}[0]'.format(tmp2, tmp))
            state_code.dedent()
            state_code.add_line('else')
            state_code.indent()
            state_code.add_line('assert(False, "Unreachable");')
            state_code.dedent()
        else:
            state_code.add_line('{} = tuple({})'.format(tmp2, tmp))
        return tmp, tmp2

    def create_operation(self, operation, operation_type, args):
        return 'create_operation_expression({}, {})'.format(operation, args)

    def exit_sequence_wildcard(self, state_code, value):
        state_code.add_line('if ({}.size() == 0) {{'.format(self._subjects[-1]))
        state_code.indent(bracket=False)
        state_code.add_line('break')
        state_code.dedent()
        state_code.add_line('{}.push_back({}.front());'.format(value, self._subjects[-1]))
        state_code.add_line('{}.pop_front();'.format(self._subjects[-1]))
        state_code.dedent()
        state_code.add_line('{}.extendleft(reversed({}))'.format(self._subjects[-1], value))
        state_code.dedent()

    def yield_final_substitution(self, state_code, pattern_index):
        renaming = self._matcher.pattern_vars[pattern_index]
        subst_name = 'subst{}'.format(self._substs)
        self.add_class_field("Substitution", subst_name)
        if any(k != v for k, v in renaming.items()):
            self.add_class_field("Substitution", "tmp_subst")
            for original, renamed in renaming.items():
                state_code.add_line('tmp_subst["{}"] = subst{}["{}"];'.format(original, self._substs, renamed))
            subst_name = 'tmp_subst'
        state_code.add_line('// {}: {}'.format(pattern_index, self._matcher.patterns[pattern_index][0]))
        state_code.add_line('yield(make_tuple({}, {}));'.format(self.final_label(pattern_index, subst_name), subst_name))

    def generate_constraints(self, state_code, constraints, transitions):
        if len(constraints) == 0:
            for i, transition in enumerate(transitions):
                removed = self._patterns - transition.patterns
                self._patterns.intersection_update(transition.patterns)
                transition_target_code = self.add_new_state(transition.target)
                transition_target_code.add_line("// Add new STATE line 582")
                #next_target_code = self.add_new_state_part(state_code)
                #next_target_code.add_line("// line 584")
                #next_method = state_code.add_transition(transition_target_code, next_target_code)
                state_code.add_pointer(transition_target_code)
                state_code.dedent(bracket=False)
                state_code = self.generate_state_code(transition.target, transition_target_code)
                self._patterns.update(removed)
        else:
            constraint_index, *remaining = constraints
            constraint, patterns = self._matcher.constraints[constraint_index]
            remaining_patterns = self._patterns - patterns
            remaining_transitions = [t for t in transitions if t.patterns & remaining_patterns]
            checked_patterns = self._patterns & patterns
            checked_transitions = [t for t in transitions if t.patterns & checked_patterns]
            other_code = self.add_new_state_part(state_code)
            if checked_patterns and checked_transitions:
                cvars = ' || '.join('(subst{1}.find("{0}") == subst{1}.end())'.format(v, self._substs) for v in constraint.variables)
                if cvars:
                    cvars += ' || '
                cexpr, call = self.constraint_repr(constraint)
                if call:
                    state_code.add_line('if ({}{}(subst{})) {{'.format(cvars, cexpr, self._substs))
                else:
                    state_code.add_line('if ({}{}) {{'.format(cvars, cexpr))
                state_code.indent(bracket=False)
                self._patterns = checked_patterns
                self.generate_constraints(state_code, other_code, remaining, checked_transitions)
                state_code.dedent(bracket=False)
                state_code.add_line('} else {')
                state_code.indent(bracket=False)
                state_code.add_line("// generate_constraints::state_code.add_pointer(other_code)")
                state_code.add_pointer(other_code)
                #state_code.dedent()
                state_code = other_code
            if remaining_patterns and remaining_transitions:
                self._patterns = remaining_patterns
                self.generate_constraints(state_code, other_code, remaining, remaining_transitions)
            self._patterns = remaining_patterns | checked_patterns
        return state_code

    def enter_global_constraint(self, constraint):
        cexpr, call = self.constraint_repr(constraint)
        if call:
            self.add_line('if ({}(subst{})) {{'.format(cexpr, self._substs))
        else:
            self.add_line('if ({}) {{'.format(cexpr))
        self.indent(bracket=False)

    def constraint_repr(self, constraint):
        if isinstance(constraint, CustomConstraint) and isinstance(constraint.constraint, type(lambda: 0)):
            src = get_short_lambda_source(constraint.constraint)
            if src is not None:
                mapping = {k: v for v, k in constraint._variables.items()}
                params = constraint._variables.keys()
                pstr = r'\b({})\b'.format('|'.join(map(re.escape, params)))
                new_src = re.sub(pstr, lambda m: 'subst{}["{}"]'.format(self._substs, constraint._variables[m[0]]), src)
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
            result = '\n{}if ({} && {})\n{}{}\n'.format(indent, m['cond1'], m['cond2'], indent2, inner)
            if m['comment']:
                result = '\n{}{}{}'.format(indent, m['comment'].strip(), result)
            return result

        count = 1
        while count > 0:
            code, count = COLLAPSE_IF_RE.subn(sub_cb, code)
        return code
