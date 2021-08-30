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


class Literal(object):

    def __init__(self, predicate, positive):
        self._predicate = predicate
        self._positive  = positive

    @property
    def predicate(self):
        return self._predicate

    def is_positive(self):
        return self._positive

    def is_negative(self):
        return not self._positive

    def is_private(self):
        return self._predicate.is_private

    @classmethod
    def positive(cls, predicate):
        return Literal(predicate, True)

    @classmethod
    def negative(cls, predicate):
        return Literal(predicate, False)

    ''' takes a list of predicates and returns a list of literals
        assumes predicates to be positive '''
    @staticmethod
    def literals_from_predicates(predicates):
        literals = []
        for pred in predicates:
            literals.append(Literal(pred, True))
        return literals

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.is_positive():
            return str(self._predicate)
        if self.is_negative() and self._predicate.name == '=':
            lhs = str(self._predicate.args[0])
            rhs = str(self._predicate.args[1])
            return '{0} != {1}'.format(lhs, rhs)
        if self.is_negative():
            return 'not {}'.format(str(self._predicate))

    def __eq__(self, other):
        return (self.is_positive() == other.is_positive()) and (self.predicate == other.predicate)

    def __hash__(self):
        return hash(str(self))

    def to_json_dict(self):
        return {
            "predicate": {
                "name": self.predicate.name,
                "arity": self.predicate.arity,
                "private": self.predicate.is_private,
                "args": self.predicate.args
            },
            "positive": self.is_positive()
        }
