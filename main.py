import sys
sys.path.append('./pypddl_parser')

from pddlparser import PDDLParser
from planner import DRTDP
from value_iteration import ValueIteration
import glob
import json
import argparse

# import cProfile, pstats, io
# from pstats import SortKey
# import re


DOMAIN_BLOCKSWORLD = 'blocks/'
DOMAIN_LOGISTICS = 'logistics/'
DOMAIN_TAXI = 'taxi/'
DOMAIN_DEPOT = 'depot/'
DOMAIN_SATELLITES = 'satellites/'
DOMAIN_ROVERS = 'rovers/'

PROB_BLOCKSWORLD = 'probBLOCKS-'
PROB_LOGISTICS = 'probLOGISTICS-'
PROB_TAXI = 'probTAXI-'
PROB_DEPOT = 'probDEPOT-'
PROB_SATELLITES = 'probSATELLITES-'
PROB_ROVERS = 'probROVERS-'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--to_json', help='create json files representing the domain and problem, '
                                          'no planning will happen with this flag',
                        action='store_true')
    parser.add_argument('--drtdp', help='use DRTDP algorithm instead of PS-RTDP which is the default',
                        action='store_true')
    parser.add_argument('--value_iteration', help='use value iteration algorithm - for small problems only',
                        action='store_true')
    parser.add_argument('--verbose', help='adds printing of progress',
                        action='store_true')
    parser.add_argument('--domain', help='specify the domain on which to plan - for example blocks',
                        required=True)
    parser.add_argument('--problem', help='specify the specific problem in the domain - for example 2-2-blue-red',
                        required=True)
    args = parser.parse_args()

    USE_DRTDP = True if args.drtdp else False
    USE_PS_RTDP = not USE_DRTDP

    pddl__path = './pypddl_parser/pddl/probabilistic/'

    domain_path = f'{args.domain}/'
    problem_path = f'prob{args.domain.upper()}-{args.problem}/'

    if args.verbose:
        print('domain received =', domain_path)
        print('problem received =', problem_path)

    domain_json_path = 'domain_jsons/' + domain_path
    problem_json_path = 'problem_jsons/' + domain_path

    domain_files = glob.glob(pddl__path + domain_path + problem_path + 'domain*.pddl')
    problem_files = glob.glob(pddl__path + domain_path + problem_path + 'problem*.pddl')

    domains = []
    problems = []

    for domain in domain_files:
        domains.append(PDDLParser.parse(domain))

    for problem in problem_files:
        problems.append(PDDLParser.parse(problem))

    if args.to_json:
        for domain in domains:
            json_file = f'{domain_json_path}/{domain.name}.json'
            with open(json_file, 'w') as file:
                json.dump(domain.to_json_dict(), file)

        for problem in problems:
            json_file = f'{problem_json_path}/{problem.name}.json'
            with open(json_file, 'w') as file:
                json.dump(problem.to_json_dict(), file)

        # no planning if to_json is on
        exit(0)

    goal_reaches = 2000
    eval_iterations = 50

    if args.value_iteration:
        planner = ValueIteration(domains, problems, goal_reaches, eval_iterations)
        planner.VERBOSE = args.verbose
    else:
        planner = DRTDP(domains, problems, goal_reaches, eval_iterations)
        planner.USE_DRTDP = USE_DRTDP
        planner.USE_PS_RTDP = USE_PS_RTDP
        planner.VERBOSE = args.verbose
        planner.heuristic_noise = 0.0
    # planner.use_heuristic = True
    #
    # cProfile.run('re.compile("planner.plan")')
    #
    # pr = cProfile.Profile()
    # pr.enable()

    # ... do something ...
    planner.plan()
    # planner.get_num_of_states()
    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
