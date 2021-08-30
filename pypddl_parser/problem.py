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

import itertools


class Problem(object):

    def __init__(self, name, domain, objects, init, goal):
        self._name = name
        self._domain = domain
        self._objects = {}
        for obj in objects:
            self._objects[obj.type] = self._objects.get(obj.type, [])
            self._objects[obj.type].append(obj)

        self._init = init
        self._goal = goal

    @property
    def name(self):
        return self._name

    @property
    def domain(self):
        return self._domain

    @property
    def objects(self):
        return self._objects.copy()

    @property
    def init(self):
        return self._init.copy()

    @property
    def goal(self):
        return self._goal.copy()

    def __str__(self):
        problem_str  = '@ Problem: {0}\n'.format(self._name)
        problem_str += '>> domain: {0}\n'.format(self._domain)
        problem_str += '>> objects:\n'
        for type, objects in self._objects.items():
            problem_str += '{0} -> {1}\n'.format(type, ', '.join(sorted(map(str, objects))))
        problem_str += '>> init:\n{0}\n'.format(', '.join(sorted(set(map(str, self._init)))))
        problem_str += '>> goal:\n{0}\n'.format(', '.join(sorted(set(map(str, self._goal)))))
        return problem_str

    def to_json_dict(self):
        objects_dict = {}
        for obj_type, terms in self.objects.items():
            objects_dict[obj_type] = [term.to_json_dict() for term in terms]
        return {
            "name": self.name,
            "domain": self.domain,
            "objects": objects_dict,
            "initial": [predicate.to_json_dict() for predicate in self.init],
            "goal": [predicate.to_json_dict() for predicate in self.goal]
        }
