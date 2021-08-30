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


class Action(object):

    def __init__(self, name, params, precond, effects, cost):
        self._name    = name
        self._params  = params
        self._precond = precond
        self._effects = effects
        self._cost = cost
        self.action_hash = None
        self.action_str = None

    @property
    def name(self):
        return self._name

    @property
    def params(self):
        return self._params[:]

    @property
    def precond(self):
        return self._precond[:]

    @property
    def effects(self):
        return self._effects[:]

    @property
    def cost(self):
        return self._cost

    def __str__(self):
        if self.action_str is None:
            operator_str  = '{0}({1})\n'.format(self._name, ', '.join(map(str, self._params)))
            operator_str += '>> precond: {0}\n'.format(', '.join(map(str, self._precond)))
            operator_str += '>> effects: {0}\n'.format(', '.join(map(str, self._effects)))
            operator_str += '>> cost: {0}\n'.format(self._cost)
            self.action_str = operator_str
        return self.action_str

    def __hash__(self):
        if self.action_hash is None:
            self.action_hash = hash(str(self))
        return self.action_hash

    def __eq__(self, other):
        return self.name == other.name and self.params == other.params

    def param_strings(self):
        '''
        :return: a list of strings representing the params of the action
        '''

        return list(map(lambda term: str(term.value), self.params))

    def to_json_dict(self):
        return {
            "name": self.name,
            "params": [term.to_json_dict() for term in self.params],
            "preconditions": [literal.to_json_dict() for literal in self.precond],
            "effects": [{"probability": prob, "effects": [literal.to_json_dict() for literal in literals]}
                        for prob, literals in self.effects],
            "cost": self.cost
        }
