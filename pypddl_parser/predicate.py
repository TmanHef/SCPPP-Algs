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


from functools import reduce

class Predicate(object):

    def __init__(self, name, private=False, args=[]):
        self._name = name
        self._args = args
        self._private = private
        self.pred_str = None

    @property
    def name(self):
        return self._name

    @property
    def args(self):
        return self._args[:]

    @property
    def arity(self):
        return len(self._args)

    @property
    def is_private(self):
        return self._private

    def _private_str(self):
        return 'private' if self._private else 'public'

    def arg_strings(self):
        str_list = list(map(lambda term: str(term), self.args))
        return '(' + reduce(lambda str1, str2: str1 + str2 + ',', str_list, '') + ')'

    def __str__(self):
        if self.pred_str is None:
            if self._name == '=':
                self.pred_str = '{0} = {1}'.format(str(self._args[0]), str(self._args[1]))
            elif self.arity == 0:
                self.pred_str = self._name
            else:
                self.pred_str = '{0},{1},({2})'.format(self._name, self._private_str(), ', '.join(map(str, self._args)))

        return self.pred_str

    def __eq__(self, other):
        return (self.name == other.name) and (self.args == other.args)

    def to_json_dict(self):
        return {
            "name": self.name,
            "arity": self.arity,
            "private": self.is_private,
            "args": [term.to_json_dict() for term in self.args]
        }
