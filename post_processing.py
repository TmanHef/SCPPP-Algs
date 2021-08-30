import json
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

RESULTS_PATH = './result_graphs/'
DOMAIN_TYPE = 'blocks'


def create_evaluations_unified_graph(multi_agent_file, single_agent_file):
    # get problem sizes and make sure they are the same
    multi_agent_prob_size = multi_agent_file[(len(DOMAIN_TYPE) + 1): (len(DOMAIN_TYPE) + 4)]
    single_agent_prob_size = single_agent_file[(len(DOMAIN_TYPE) + 1): (len(DOMAIN_TYPE) + 4)]

    if multi_agent_prob_size == single_agent_prob_size:

        file_path = '{0}{1}/{2}/'.format(RESULTS_PATH, DOMAIN_TYPE, 'evaluations')
        multi_agent_file = file_path + multi_agent_file
        single_agent_file = file_path + single_agent_file

        with open(multi_agent_file) as f1, open(single_agent_file) as f2:
            multi_agent_evals = json.load(f1)
            single_agent_evals = json.load(f2)

            if len(multi_agent_evals) == len(single_agent_evals):

                print('... saving evaluations figure ...')
                x_axis = list(range(len(multi_agent_evals)))
                plt.plot(x_axis, multi_agent_evals, 'C0', label='multi agent')
                plt.plot(x_axis, single_agent_evals, 'C1', label='single agent')
                plt.legend()

                plt.title('multi VS single agent evaluations')
                plt.xlabel('evaluation number')
                plt.ylabel('average sum of costs')
                out_file_path = '{0}{1}/{2}/'.format(RESULTS_PATH, DOMAIN_TYPE, 'comparisons')
                file_type = 'png'
                file_name = out_file_path + 'multi_vs_single_' + multi_agent_prob_size + '.' + file_type

                plt.savefig(file_name, format=file_type, dpi=400)
                plt.cla()

            else:
                print('number of evaluations not is not the same for both files!')

    else:
        print('multi agent and single agent problem sizes are not equal')


def combine_results_jsons(drtdp_json, psrtdp_json, vi_json):
    """
    takes overall results jsons and combines them to one json
    :param drtdp_json a json for drtdp overall results
    :param psrtdp_json a json for ps-rtdp overall results
    :param vi_json a json for value iteration overall results
    :return combined json
    """
    vi_cost = vi_json['best_cost'] if 'best_cost' in vi_json else None
    vi_time = '{0:.3f}'.format(vi_json['planning_time']) if 'planning_time' in vi_json else None
    return {
        "domain": drtdp_json['domain'],
        "actions": drtdp_json['actions'],
        "facts": drtdp_json['facts'],
        "num_agents": drtdp_json['num_agents'],
        "best_cost": (drtdp_json['best_cost'], psrtdp_json['best_cost'], vi_cost),
        "messages": ('{0:.3f}'.format(drtdp_json['messages']/10000),
                     '{0:.3f}'.format(psrtdp_json['messages']/10000)),
        "expansions": ('{0:.3f}'.format(drtdp_json['expansions']/10000),
                       '{0:.3f}'.format(psrtdp_json['expansions']/10000)),
        "trajectories": (drtdp_json['trajectories'], psrtdp_json['trajectories']),
        "restarts": (drtdp_json['restarts'], psrtdp_json['restarts']),
        "planning_time": ('{0:.3f}'.format(drtdp_json['planning_time']),
                          '{0:.3f}'.format(psrtdp_json['planning_time']),
                          vi_time)
    }


def combine_results(json_tuple_list):
    combined_list = []
    for drtdp_json_file, psrtdp_json_file, vi_json_file in json_tuple_list:
        with open(drtdp_json_file) as f1, open(psrtdp_json_file) as f2, open(vi_json_file) as f3:
            drtdp_json = json.load(f1)
            psrtdp_json = json.load(f2)
            vi_json = json.load(f3)
            combined = combine_results_jsons(drtdp_json, psrtdp_json, vi_json)
            combined_list.append(combined)
    return combined_list


def wrap_in_latex_bold(value):
    return '\\textbf{' + str(value) + '}'


def get_smaller_of_2_in_bold_latex(val1, val2):
    if val1 < val2:
        bold_latex_str = f'{wrap_in_latex_bold(val1)}\t& {val2}'
    elif val1 > val2:
        bold_latex_str = f'{val1}\t& {wrap_in_latex_bold(val2)}'
    else:
        bold_latex_str = f'{wrap_in_latex_bold(val1)}\t& ' \
                         f'{wrap_in_latex_bold(val2)}'
    return bold_latex_str


def get_smaller_of_3_in_bold_latex(val1, val2, val3):
    if val1 is None:
        return f'$\\times$\t& {get_smaller_of_2_in_bold_latex(val2, val3)}'

    if val1 < val2 and val1 < val3:
        bold_latex_str = f'{wrap_in_latex_bold(val1)}\t& {val2}\t& {val3}'
    elif val2 < val1 and val2 < val3:
        bold_latex_str = f'{val1}\t& {wrap_in_latex_bold(val2)}\t& {val3}'
    elif val3 < val1 and val3 < val2:
        bold_latex_str = f'{val1}\t& {val2}\t& {wrap_in_latex_bold(val3)}'
    elif val1 == val2 == val3:
        bold_latex_str = f'{wrap_in_latex_bold(val1)}\t& {wrap_in_latex_bold(val2)}\t& {wrap_in_latex_bold(val3)}'
    elif val1 == val2:
        bold_latex_str = f'{wrap_in_latex_bold(val1)}\t& {wrap_in_latex_bold(val2)}\t& {val3}'
    elif val1 == val3:
        bold_latex_str = f'{wrap_in_latex_bold(val1)}\t& {val2}\t& {wrap_in_latex_bold(val3)}'
    elif val2 == val3:
        bold_latex_str = f'{val1}\t& {wrap_in_latex_bold(val2)}\t& {wrap_in_latex_bold(val3)}'
    else:
        print('get_smaller_in_bold_latex - unexpected case: val1 =', val1, 'val2 =', val2, 'val3 =', val3)
        exit(1)
    return bold_latex_str


def combined_to_latex(combined_dict_list):
    now = datetime.now()
    time_str = now.strftime("%d-%m-%Y-%H-%M")
    out_path = f'latex_results/{time_str}.txt'
    with open(out_path, 'x') as out_file:
        table1_lines = []
        table2_lines = []
        prev_domain = ''
        domain_counter = 0
        for combined in combined_dict_list:
            domain_name = combined["domain"][:combined["domain"].index('-')]
            if not domain_name == prev_domain:
                domain_counter = 1
            else:
                domain_counter += 1

            domain_str = f'{domain_name}-{domain_counter}'
            vi_cost = float(combined['best_cost'][2]) if not combined['best_cost'][2] is None else None
            cost_tup_latex = get_smaller_of_3_in_bold_latex(vi_cost,
                                                            float(combined['best_cost'][0]),
                                                            float(combined['best_cost'][1]))

            vi_time = float(combined['planning_time'][2]) if not combined['planning_time'][2] is None else None
            time_tup_latex = get_smaller_of_3_in_bold_latex(vi_time,
                                                            float(combined['planning_time'][0]),
                                                            float(combined['planning_time'][1]))

            # \textbf{blocks-1}	& 1 & 2 & 3		& 1 & 1 & 1	    & 0.1 & 0.1 & 0.1		\\ \hline % blocks-2-2-blue-red

            line1 = f'{wrap_in_latex_bold(domain_str)}\t\t\t& {combined["actions"]}\t\t& {combined["facts"]}\t\t' \
                    f' & {combined["num_agents"]}\t\t& {cost_tup_latex}\t\t& {time_tup_latex}\t\t\t\\\\ \\hline'

            # currently i have 5 problems in each domain so on the fifth domain add \\hline
            line1 += f'\\hline %{combined["domain"]}\n' if domain_counter % 5 == 0 else f' % {combined["domain"]}\n'

            cost_tup_latex = get_smaller_of_2_in_bold_latex(float(combined['best_cost'][0]),
                                                            float(combined['best_cost'][1]))
            expansions_tup_latex = get_smaller_of_2_in_bold_latex(float(combined['expansions'][0]),
                                                                  float(combined['expansions'][1]))
            messages_tup_latex = get_smaller_of_2_in_bold_latex(float(combined['messages'][0]),
                                                                float(combined['messages'][1]))
            line2 = f'{wrap_in_latex_bold(domain_str)}\t\t\t& {cost_tup_latex}\t\t& {expansions_tup_latex}\t\t& ' \
                    f'{messages_tup_latex}\t\t& ({combined["trajectories"][0]} / {combined["trajectories"][1]} + ' \
                    f'{combined["restarts"][1]})\t\t\t\\\\ \\hline'

            # currently i have 5 problems in each domain so on the fifth domain add \\hline
            line2 += f'\\hline %{combined["domain"]}\n' if domain_counter % 5 == 0 else f' % {combined["domain"]}\n'

            table1_lines.append(line1)
            table2_lines.append(line2)

            prev_domain = domain_name

        print('writing to file', out_path)
        out_file.writelines(table1_lines + table2_lines)
        print('done')


def combined_vi_to_latex(dict_list):
    now = datetime.now()
    time_str = now.strftime("%d-%m-%Y-%H-%M")
    out_path = f'latex_results/VI-{time_str}.txt'
    with open(out_path, 'x') as out_file:
        table_lines = []
        for res_dict in dict_list:
            domain_name = res_dict["domain"][:res_dict["domain"].index('-')]
            domain_str = f'{domain_name}-1'

            # domain & cost & iterations & messages & time
            time_str = '{0:.3f}'.format(res_dict['planning_time'])
            messages_str = '{0:.3f}'.format(res_dict['messages']/10000)
            line = f'{wrap_in_latex_bold(domain_str)}\t\t\t& {res_dict["best_cost"]}\t\t& ' \
                   f'{int(res_dict["iterations"])}\t\t & {messages_str}\t\t& {time_str}\t\t\t\\\\ ' \
                   f'\\hline % {res_dict["domain"]}\n'

            table_lines.append(line)

        print('writing to file', out_path)
        out_file.writelines(table_lines)
        print('done')


if __name__ == '__main__':

    domains = ['blocks', 'depot', 'logistics', 'rovers']

    path_tuples = []
    for domain in domains:
        overall_path = f'result_graphs/{domain}/overall/'

        if domain == "blocks":
            problems = ['2-2-blue-red', '3-3-blue-red', '4-3-blue-red', '5-2-blue-red', '6-2-blue-red']
        elif domain == "logistics":
            problems = ['1-3',  '2-3', '2-4', '2-5', '3-4']
        elif domain == "depot":
            problems = ['4-3', '5-3', '5-4', '7-5', '8-5']
        elif domain == "rovers":
            problems = ['2-2', '3-2', '3-3', '4-3', '3-4']

        for problem in problems:
            problem_str = f'{overall_path}{domain}-{problem}'
            drtdo_file = f'{problem_str}DRTDP.json'
            psrtdp_file = f'{problem_str}PS-RTDP.json'
            vi_file = f'{problem_str}VI.json'
            path_tuples.append((drtdo_file, psrtdp_file, vi_file))

    combined_to_latex(combine_results(path_tuples))
