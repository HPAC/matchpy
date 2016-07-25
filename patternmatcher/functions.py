# -*- coding: utf-8 -*-
from patternmatcher.expressions import Variable, Operation, Arity
from patternmatcher.constraints import Constraint, CustomConstraint, EqualVariablesConstraint, MultiConstraint

def linearize(expression, variables=None, constraints=None):
    if variables is None:
        variables = {}
        constraints = {}
        names = set(expression.variables.keys())

        for (name, count) in expression.variables.items():
            variables[name] = [name]

            i = 2
            for _ in range(count - 1):
                newName = name + '_' + str(i)
                while newName in names:
                    i += 1
                    newName = name + '_' + str(i)

                variables[name].append(newName)
                names.add(newName)
                i += 1

            if len(variables[name]) > 1:
                constraints[name] = EqualVariablesConstraint(*variables[name])

    # TODO: Make non-mutating  
    if isinstance(expression, Variable):
        name = expression.name
        expression.name  = variables[name].pop(0)

        if len(variables[name]) == 0 and name in constraints:
            if expression.constraint:
                if not isinstance(expression.constraint, Constraint):
                    expression.constraint = CustomConstraint(expression.constraint)
                
                expression.constraint = MultiConstraint(expression.constraint, constraints[name])
            else:
                expression.constraint = constraints[name]

        linearize(expression.expression, variables, constraints)
    elif isinstance(expression, Operation):
        for operand in expression.operands:
            linearize(operand, variables, constraints)

if __name__ == '__main__':
    f = Operation.new('f', Arity.polyadic)
    g = Operation.new('g', Arity.variadic)
    expr = f(Variable.dot('x'), f(Variable.dot('y'), Variable.dot('x')), g(Variable.dot('y')))
    linearize(expr)
    print(expr)