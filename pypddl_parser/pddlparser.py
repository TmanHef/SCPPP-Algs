# This file is part of pypddl_parser.

# pypddl_parser is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# pypddl_parser is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with pypddl_parser.  If not, see <http://www.gnu.org/licenses/>.


from ply import lex
from ply import yacc

from pypddl_parser.term      import Term
from pypddl_parser.literal   import Literal
from pypddl_parser.predicate import Predicate
from pypddl_parser.action    import Action
from pypddl_parser.domain    import Domain
from pypddl_parser.problem   import Problem


tokens = (
    'NAME',
    'VARIABLE',
    'PROBABILITY',
    'COST',
    'LPAREN',
    'RPAREN',
    'HYPHEN',
    'EQUALS',
    'DEFINE_KEY',
    'DOMAIN_KEY',
    'REQUIREMENTS_KEY',
    'STRIPS_KEY',
    'EQUALITY_KEY',
    'TYPING_KEY',
    'PROBABILISTIC_EFFECTS_KEY',
    'TYPES_KEY',
    'PREDICATES_KEY',
    'ACTION_KEY',
    'PARAMETERS_KEY',
    'PRECONDITION_KEY',
    'EFFECT_KEY',
    'AND_KEY',
    'NOT_KEY',
    'PROBABILISTIC_KEY',
    'PROBLEM_KEY',
    'OBJECTS_KEY',
    'INIT_KEY',
    'GOAL_KEY',
    'FACTORED_PRIVACY',
    'PRIVATE_KEY',
    'COST_KEY',
    'CONSTANTS_KEY'
)


t_LPAREN = r'\('
t_RPAREN = r'\)'
t_HYPHEN = r'\-'
t_EQUALS = r'='

t_ignore = ' \t'

reserved = {
    'define'                    : 'DEFINE_KEY',
    'domain'                    : 'DOMAIN_KEY',
    ':requirements'             : 'REQUIREMENTS_KEY',
    ':strips'                   : 'STRIPS_KEY',
    ':equality'                 : 'EQUALITY_KEY',
    ':typing'                   : 'TYPING_KEY',
    ':probabilistic-effects'    : 'PROBABILISTIC_EFFECTS_KEY',
    ':types'                    : 'TYPES_KEY',
    ':predicates'               : 'PREDICATES_KEY',
    ':action'                   : 'ACTION_KEY',
    ':parameters'               : 'PARAMETERS_KEY',
    ':precondition'             : 'PRECONDITION_KEY',
    ':effect'                   : 'EFFECT_KEY',
    'and'                       : 'AND_KEY',
    'not'                       : 'NOT_KEY',
    'probabilistic'             : 'PROBABILISTIC_KEY',
    'problem'                   : 'PROBLEM_KEY',
    ':domain'                   : 'DOMAIN_KEY',
    ':objects'                  : 'OBJECTS_KEY',
    ':init'                     : 'INIT_KEY',
    ':goal'                     : 'GOAL_KEY',
    ':factored-privacy'         : 'FACTORED_PRIVACY',
    ':private'                  : 'PRIVATE_KEY',
    ':cost'                     : 'COST_KEY',
    ':constants'                : 'CONSTANTS_KEY'
}

private_predicates = []
public_predicates = []
private_constants = []
domain_constants = []
object_types = {}
types_hierarchy = {}
domain_constants_types = {}


def t_KEYWORD(t):
    r':?[a-zA-z_][a-zA-Z_0-9\-]*'
    t.type = reserved.get(t.value, 'NAME')
    return t


def t_NAME(t):
    r'[a-zA-z_][a-zA-Z_0-9\-]*'
    return t


def t_VARIABLE(t):
    r'\?[a-zA-z_][a-zA-Z_0-9\-]*'
    return t


def t_PROBABILITY(t):
    r'[0-1]\.\d+'
    t.value = float(t.value)
    return t


def t_COST(t):
    r'[0-9]+'
    t.value = int(t.value)
    return t


def t_newline(t):
    r'\n+'
    t.lineno += len(t.value)


def t_error(t):
    print("Error: illegal character '{0}'".format(t.value[0]))
    t.lexer.skip(1)


# build the lexer
lex.lex()


def p_pddl(p):
    """pddl : domain
            | problem"""
    p[0] = p[1]


def p_domain(p):
    """domain : LPAREN DEFINE_KEY domain_def require_def types_def predicates_def action_def_lst RPAREN
              | LPAREN DEFINE_KEY domain_def require_def types_def constants_def predicates_def action_def_lst RPAREN"""
    if len(p) == 9:
        p[0] = Domain(p[3], p[4], p[5], types_hierarchy, p[6], p[7])
    elif len(p) == 10:
        p[0] = Domain(p[3], p[4], p[5], types_hierarchy, p[7], p[8])


def p_problem(p):
    """problem : LPAREN DEFINE_KEY problem_def domain_def objects_def init_def goal_def RPAREN"""
    p[0] = Problem(p[3], p[4], p[5], p[6], p[7])


def p_domain_def(p):
    """domain_def : LPAREN DOMAIN_KEY NAME RPAREN"""
    p[0] = p[3]


def p_problem_def(p):
    """problem_def : LPAREN PROBLEM_KEY NAME RPAREN"""
    p[0] = p[3]


def p_constants_def(p):
    """constants_def : LPAREN CONSTANTS_KEY typed_constants_lst RPAREN"""
    domain_constants = p[3]
    for term in domain_constants:
        domain_constants_types[term.value.value] = term.type


def p_objects_def(p):
    """objects_def : LPAREN OBJECTS_KEY typed_constants_lst RPAREN
                   | LPAREN OBJECTS_KEY typed_constants_lst LPAREN PRIVATE_KEY private_typed_constants_lst RPAREN RPAREN"""
    if len(p) == 5:
        p[0] = p[3]
    elif len(p) == 9:
        p[0] = p[3] + p[6]

    # add each object to a dict mapping it's name to it's type
    for obj in p[0]:
        object_types[obj.value.value] = obj.type


def p_init_def(p):
    """init_def : LPAREN INIT_KEY LPAREN AND_KEY ground_predicates_lst RPAREN RPAREN
                | LPAREN INIT_KEY ground_predicates_lst RPAREN"""
    if len(p) == 5:
        p[0] = p[3]
    elif len(p) == 8:
        p[0] = p[5]


def p_goal_def(p):
    """goal_def : LPAREN GOAL_KEY LPAREN AND_KEY ground_predicates_lst RPAREN RPAREN"""
    p[0] = p[5]


def p_require_def(p):
    """require_def : LPAREN REQUIREMENTS_KEY require_key_lst RPAREN"""
    p[0] = p[3]


def p_require_key_lst(p):
    """require_key_lst : require_key require_key_lst
                       | require_key"""
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [p[1]] + p[2]


def p_require_key(p):
    """require_key : STRIPS_KEY
                   | EQUALITY_KEY
                   | TYPING_KEY
                   | PROBABILISTIC_EFFECTS_KEY
                   | FACTORED_PRIVACY
                   """
    p[0] = str(p[1])


def p_types_def(p):
    """types_def : LPAREN TYPES_KEY sub_types_lst RPAREN"""

    p[0] = p[3]


def p_sub_types_lst(p):
    """sub_types_lst : names_lst
                     | names_lst HYPHEN NAME
                     | names_lst HYPHEN NAME sub_types_lst"""
    if len(p) == 2:
        p[0] = p[1]
    if len(p) == 4:
        p[0] = p[1]
        types_hierarchy[p[3]] = p[1]
    if len(p) == 5:
        p[0] = p[1] + p[4]
        types_hierarchy[p[3]] = p[1]


def p_predicates_def(p):
    """predicates_def : LPAREN PREDICATES_KEY predicate_def_lst RPAREN"""
    p[0] = p[3]


def p_predicate_def_lst(p):
    """predicate_def_lst : predicate_def predicate_def_lst
                         | predicate_def
                         | LPAREN PRIVATE_KEY private_predicate_def_lst RPAREN"""
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [p[1]] + p[2]
    elif len(p) == 5:
        p[0] = p[3]


def p_predicate_def(p):
    """predicate_def : LPAREN NAME typed_variables_lst RPAREN
                     | LPAREN NAME RPAREN"""
    if len(p) == 4:
        p[0] = Predicate(p[2], False)
    elif len(p) == 5:
        p[0] = Predicate(p[2], False, p[3])

    public_predicates.append(p[2])


def p_private_predicate_def_lst(p):
    """private_predicate_def_lst : private_predicate_def private_predicate_def_lst
                                 | private_predicate_def"""
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [p[1]] + p[2]


def p_private_predicate_def(p):
    """private_predicate_def : LPAREN NAME typed_variables_lst RPAREN
                             | LPAREN NAME RPAREN"""
    if len(p) == 4:
        p[0] = Predicate(p[2], True)
    elif len(p) == 5:
        p[0] = Predicate(p[2], True, p[3])

    private_predicates.append(p[2])


def p_action_def_lst(p):
    """action_def_lst : action_def action_def_lst
                      | action_def"""
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [p[1]] + p[2]
    # print('got actions', p[0])


def p_action_def(p):
    """action_def : LPAREN ACTION_KEY NAME parameters_def action_def_body RPAREN"""
    p[0] = Action(p[3], p[4], p[5][0], p[5][1], p[5][2])


def p_parameters_def(p):
    """parameters_def : PARAMETERS_KEY LPAREN typed_variables_lst RPAREN
                      | PARAMETERS_KEY LPAREN RPAREN"""
    if len(p) == 4:
        p[0] = []
    elif len(p) == 5:
        p[0] = p[3]
    # print('got params', len(p[0]))


def p_action_def_body(p):
    """action_def_body : precond_def effects_def cost_def"""
    p[0] = (p[1], p[2], p[3])


def p_precond_def(p):
    """precond_def : PRECONDITION_KEY LPAREN AND_KEY literals_lst RPAREN
                   | PRECONDITION_KEY literal"""
    if len(p) == 3:
        p[0] = [p[2]]
    elif len(p) == 6:
        p[0] = p[4]
    # print('got precond', len(p[0]))


def p_effects_def(p):
    """effects_def : EFFECT_KEY LPAREN AND_KEY effects_lst RPAREN
                   | EFFECT_KEY effect
                   | EFFECT_KEY LPAREN PROBABILISTIC_KEY prob_effects_lst RPAREN"""
    if len(p) == 3:
        p[0] = [(1.0, [p[2]])]
    elif len(p) == 6:
        if p[3] == 'and':
            p[0] = [(1.0, p[4])]
        else:
            p[0] = p[4]
    # print('got effects', p[0])


def p_cost_def(p):
    """cost_def : COST_KEY COST"""
    p[0] = p[2]


def p_prob_effects_lst(p):
    """prob_effects_lst : PROBABILITY effect prob_effects_lst
                        | PROBABILITY LPAREN AND_KEY effects_lst RPAREN prob_effects_lst
                        | PROBABILITY effect
                        | PROBABILITY LPAREN AND_KEY effects_lst RPAREN"""
    if len(p) == 3:
        p[0] = [(p[1], [p[2]])]
    elif len(p) == 4:
        p[0] = [(p[1], [p[2]])] + [p[3]]
    elif len(p) == 6:
        p[0] = [(p[1], p[4])]
    elif len(p) == 7:
        if type(p[6]) == type([]):
            p[0] = [(p[1], p[4])] + p[6]
        else:
            p[0] = [(p[1], p[4])] + [p[6]]


def p_effects_lst(p):
    """effects_lst : effect effects_lst
                   | effect"""
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [p[1]] + p[2]


def p_effect(p):
    """effect : literal"""
    p[0] = p[1]


def p_literals_lst(p):
    """literals_lst : literal literals_lst
                    | literal"""
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [p[1]] + p[2]


def p_literal(p):
    """literal : LPAREN NOT_KEY predicate RPAREN
               | LPAREN NOT_KEY ground_predicate RPAREN
               | LPAREN NOT_KEY mixed_predicate RPAREN
               | predicate
               | ground_predicate
               | mixed_predicate"""
    if len(p) == 2:
        p[0] = Literal.positive(p[1])
    elif len(p) == 5:
        p[0] = Literal.negative(p[3])


def p_ground_predicates_lst(p):
    """ground_predicates_lst : ground_predicate ground_predicates_lst
                             | ground_predicate"""
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [p[1]] + p[2]


def p_predicate(p):
    """predicate : LPAREN NAME variables_lst RPAREN
                 | LPAREN EQUALS VARIABLE VARIABLE RPAREN
                 | LPAREN NAME RPAREN"""
    private = True if p[2] in private_predicates else False
    if len(p) == 4:
        p[0] = Predicate(p[2], private)
    elif len(p) == 5:
        p[0] = Predicate(p[2], private, p[3])
    elif len(p) == 6:
        p[0] = Predicate('=', private, [p[3], p[4]])


def p_ground_predicate(p):
    """ground_predicate : LPAREN NAME constants_lst RPAREN"""
    if len(p) == 4:
        private = True if p[2] in private_predicates else False
        p[0] = Predicate(p[2], private)
    elif len(p) == 5:

        # add types to the args according to objects mapping
        typed_args = []
        private = False
        for term in p[3]:
            # print('ground predicate name =', p[2])
            # print('got constants_lst:', str(p[3]))
            term_type = object_types.get(term.value, None)
            if term_type is None:
                term_type = domain_constants_types[term.value]
                # print('ground predicate arg', term.value, 'has type', term_type)
            typed_term = Term(name=term.value, type=term_type, value=term.value, private=term.private)
            typed_args.append(typed_term)
            if typed_term.private:
                private = True
        p[0] = Predicate(p[2], private, typed_args)
        # print('created predicate', str(p[0]))


def p_mixed_predicate(p):
    """mixed_predicate : LPAREN NAME constants_lst variables_lst RPAREN"""
    private = True if p[2] in private_predicates else False
    if len(p) == 4:
        p[0] = Predicate(p[2], private)
    elif len(p) == 6:

        # add types to the args according to objects mapping
        typed_args = []
        for term in p[3]:
            term_type = object_types.get(term.value, None)
            if term_type is None:
                term_type = domain_constants_types[term.value]
                # print('mixed predicate arg', term.value, 'has type', term_type)
            typed_term = Term(name=term.value, type=term_type, value=term.value, private=term.private)
            typed_args.append(typed_term)
            if typed_term.private:
                private = True
        p[0] = Predicate(p[2], private, typed_args+p[4])


def p_typed_constants_lst(p):
    """typed_constants_lst : constants_lst HYPHEN type typed_constants_lst
                           | constants_lst HYPHEN type"""
    if len(p) == 4:
        p[0] = [ Term.constant(value, False, p[3]) for value in p[1] ]
    elif len(p) == 5:
        p[0] = [ Term.constant(value, False, p[3]) for value in p[1] ] + p[4]


def p_private_typed_constants_lst(p):
    """private_typed_constants_lst : private_constants_lst HYPHEN type private_typed_constants_lst
                                   | private_constants_lst HYPHEN type"""
    if len(p) == 4:
        p[0] = [ Term.constant(value, True, p[3]) for value in p[1] ]
    elif len(p) == 5:
        p[0] = [ Term.constant(value, True, p[3]) for value in p[1] ] + p[4]


def p_private_constants_lst(p):
    """private_constants_lst : constant private_constants_lst
                             | constant"""
    if len(p) == 2:
        p[0] = [Term.constant(p[1], True)]
        private_constants.append(p[1])
    elif len(p) == 3:
        p[0] = [Term.constant(p[1], True)] + p[2]


def p_typed_variables_lst(p):
    """typed_variables_lst : variables_lst HYPHEN type typed_variables_lst
                           | variables_lst HYPHEN type"""
    if len(p) == 4:
        p[0] = [ Term.variable(name, p[3]) for name in p[1] ]
    elif len(p) == 5:
        p[0] = [ Term.variable(name, p[3]) for name in p[1] ] + p[4]


def p_constants_lst(p):
    """constants_lst : constant constants_lst
                     | constant"""

    private = True if p[1] in private_constants else False
    if len(p) == 2:
        p[0] = [ Term.constant(p[1], private) ]
    elif len(p) == 3:
        p[0] = [ Term.constant(p[1], private) ] + p[2]


def p_variables_lst(p):
    """variables_lst : VARIABLE variables_lst
                     | VARIABLE"""
    if len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [p[1]] + p[2]


def p_names_lst(p):
    """names_lst : NAME names_lst
                 | NAME"""
    if len(p) == 1:
        p[0] = []
    elif len(p) == 2:
        p[0] = [p[1]]
    elif len(p) == 3:
        p[0] = [p[1]] + p[2]


def p_type(p):
    """type : NAME"""
    p[0] = p[1]


def p_constant(p):
    """constant : NAME"""
    p[0] = p[1]


def p_error(p):
    print("Error: syntax error when parsing '{}'".format(p))


# build parser
yacc.yacc()


class PDDLParser(object):

    @classmethod
    def parse(cls, filename):
        data = cls.__read_input(filename)
        return yacc.parse(data)

    @classmethod
    def __read_input(cls, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            data = ''
            for line in file:
                line = line.rstrip().lower()
                line = cls.__strip_comments(line)
                data += '\n' + line
        return data

    @classmethod
    def __strip_comments(cls, line):
        pos = line.find(';')
        if pos != -1:
            line = line[:pos]
        return line
