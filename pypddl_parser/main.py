# This file is part of pypddl-PDDLParser.

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


import argparse

from pddlparser import PDDLParser

#
# def parse():
#     usage = 'python3 main.py <DOMAIN> <INSTANCE>'
#     description = 'pypddl_parser is a PDDL parser built on top of ply.'
#     parser = argparse.ArgumentParser(usage=usage, description=description)
#
#     parser.add_argument('domain',  type=str, help='path to PDDL domain file')
#     parser.add_argument('problem', type=str, help='path to PDDL problem file')
#
#     return parser.parse_args()
#
#
# if __name__ == '__main__':
#     # args = parse()
#     #
#     # domain  = PDDLParser.parse(args.domain)
#     # problem = PDDLParser.parse(args.problem)
#
#     domain_file = './pddl/blocksworld/domain.pddl'
#     problem_file = './pddl/blocksworld/problems/probBLOCKS-04-0.pddl'
#
#     domain  = PDDLParser.parse(domain_file)
#     problem = PDDLParser.parse(problem_file)
#
#     problem.
#
#     print(domain)
#     print(problem)
