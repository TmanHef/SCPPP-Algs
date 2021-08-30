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


class Term(object):

    def __init__(self, **kwargs):
        self._name  = kwargs.get('name',  None)
        self._type  = kwargs.get('type',  None)
        self._value = kwargs.get('value', None)
        self._private = kwargs.get('private', False)
        self.term_hash = None
        self.term_str = None

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def value(self):
        return self._value

    @property
    def private(self):
        return self._private

    def set_private(self, is_private):
        self._private = is_private

    def is_variable(self):
        return self._name is not None

    def is_typed(self):
        return self._type is not None

    def is_constant(self):
        return self._value is not None

    def is_private(self):
        return self._private

    @classmethod
    def variable(cls, name, type=None):
        return Term(name=name, type=type)

    @classmethod
    def constant(cls, value, private, type=None):
        return Term(value=value, private=private, type=type)

    def _private_str(self):
        return 'private' if self._private else 'public'

    def __str__(self):
        if self.term_str is None:
            if self.is_variable() and self.is_typed():
                self.term_str = '{0} - {1}'.format(self._name, self._type)
            elif self.is_variable():
                self.term_str = '{0}'.format(self._name)
            elif self.is_constant() and self.is_typed():
                self.term_str = '{0} - {1}'.format(self._value, self._type)
            elif self.is_constant():
                self.term_str = '{0}'.format(self._value)
            # return '{0} - {1}'.format(self._name, self._type)

        return self.term_str

    def __eq__(self, other):
        return (self.name == other.name) and (self.is_private() == other.is_private()) and \
               (self.value == other.value) and (self.type == other.type)

    def __hash__(self):
        if self.term_hash is None:
            self.term_hash = hash(str(self))
        return self.term_hash

    def to_json_dict(self):
        return {
            "name": self.name,
            "type": self.type,
            "private": self.private
        }
