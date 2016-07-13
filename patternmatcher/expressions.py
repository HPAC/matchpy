# -*- coding: utf-8 -*-

class Expression(object):
    pass

class Operation(Expression):
    pass

class Atom(Expression):
    pass

class Symbol(Atom):
    pass

class Wildcard(Atom):
    pass
