from message import PUBLIC_STATE_MSG
from message import BACK_UPDATE_MSG
from message import QVALUE_REQ_MSG
from message import QVALUE_ANS_MSG
from message import Message
from agent import Agent
from state import State
from predicate import Predicate
from termcolor import colored
from functools import reduce
from copy import deepcopy
import matplotlib.pyplot as plt
import time
import random
import json
import csv
import itertools


class DRTDP(object):

    RESULTS_PATH = './result_graphs/'

    def __init__(self, domains, problems, iterations, eval_iterations):
        self._domains = domains
        self._problems = problems

        # a dictionary mapping public states by their hash (public states have no private literals)
        self.states = {}
        self._current_agent = -1
        self.iterations = iterations
        self.eval_iterations = eval_iterations
        self.current_iteration = 1

        # for monitoring
        self.goal_trajectories = {}
        self.best_goal_trajectory = None
        self.best_goal_trajectory_cost = float('inf')
        self.goal_time_list = []
        self.evaluations = []
        self.agent_colors = ['red', 'green', 'blue',  'cyan', 'white', 'yellow', 'grey', 'magenta']
        self.last_start_time = None
        self.first_start_time = None
        self.eval_time = 0
        self.best_eval_actions = None
        self.best_reached_goal = False
        self.bad_trajectory_counts = []
        self.current_bad_trajectory_count = 0
        self.USE_DRTDP = False
        self.USE_PS_RTDP = True
        self.VERBOSE = False
        self.num_facts = 0

        # always use the same seed for easier debugging
        self.random = random
        self.random.seed(100)
        self.heuristic_noise = 0.0

        # do not use heuristic by default
        self.use_heuristic = False

        agents = []
        agent_initial_keys = {}
        agent_initial_literals = {}
        problems = list(sorted(problems, key=lambda prob: prob.name))
        for problem in problems:
            filtered = list(filter(lambda domain: domain.name == problem.domain, domains))
            if len(filtered) == 0:
                print('DRTDP -- no domain with the name: ', problem.name)
                exit(1)
            agent, initial_literals, initial_private_keys, goal_literals = Agent.build_agent(filtered[0], problem, self)
            agents.append(agent)
            agent_initial_keys[agent.id] = initial_private_keys
            agent_initial_literals[agent.id] = initial_literals

        self._agents = agents

        # add goal state to states dict
        goal_state = State(goal_literals, {}, self)
        goal_hash = hash(goal_state)
        if goal_hash not in self.states:
            goal_state.id = len(self.states)
            self.states[goal_hash] = goal_state

        for agent in agents:

            # in case agents have different initial states - todo - not necessary, since private keys of all other
            #  agents are considered part of the state, the hash will be the same for all agents
            state_hash = State.get_hash(agent_initial_literals[agent.id], agent_initial_keys)
            initial_state = self.states.get(state_hash, None)

            if initial_state is None:
                initial_state = State(agent_initial_literals[agent.id], agent_initial_keys, self, True)

            if state_hash not in self.states:
                initial_state.id = len(self.states)
                self.states[state_hash] = initial_state

            agent.initial_state = initial_state
            agent.goal_state = goal_state

        # a dictionary mapping agents to a list of their messages
        self._messages = {agent.id: [] for agent in agents}

        # for monitoring
        self.initial_Q_values = {agent.id: [] for agent in agents}
        self.agent_traj_pass = {agent.id: 0 for agent in agents}
        self.agent_messages_count = {agent.id: [] for agent in agents}
        self.expansion_counts = {agent.id: [] for agent in agents}
        self.q_tables = {agent.id: [] for agent in agents}


    @property
    def agents(self):
        return self._agents

    # for now select by round robin
    def select_agent(self):
        next_agent = (self._current_agent + 1) % len(self._agents)

        # if this is the first time selecting agent - add he first trajectory from the agent's initial state
        # this is for single trajectory at a time - we need to start
        if self._current_agent == -1:
            self.agents[next_agent].create_new_trajectory(self.agents[next_agent].initial_state)

        self._current_agent = next_agent
        return self._agents[next_agent]

    def process_messages(self, agent):
        '''go over all messages in the queue,
           for each message checks it's type (PUBLIC_STATE_MSG / GOAL_STATE_MSG)
           and act accordingly'''

        should_handle_q_ans = False
        BEST_ANS_Q = 'best Q value answer'
        BEST_ANS_ID = 'best Q value answering agent id'
        BEST_ANS_MSG = 'best Q value answer message'

        # for handling QValue requests
        # dict mapping states to their best ans data {state_hash: {BEST_ANS_Q: float, BEST_ANS_ID: int,
        # BEST_ANS_MSG: Message}}
        q_val_ans_dict = {}

        for message in self._messages[agent.id]:
            if message.type == QVALUE_REQ_MSG:
                agent.current_message_count[QVALUE_REQ_MSG] += 1

                # get agent's best Q for the message.state
                best_action, _ = agent.greedy_action(message.state)

                if best_action is None:
                    agent_best_q = float('inf')
                else:
                    agent_best_q = agent.Q_values[message.state][best_action]

                q_val_msg = Message(QVALUE_ANS_MSG, message.state, agent.id, message.trajectory_key,
                                    message.gen_agent_id, agent_best_q, prev_state=message.prev_state,
                                    prev_action=best_action)

                self.add_message(q_val_msg)

            # agent got answer for a QVALUE_REQ_MSG -
            elif message.type == QVALUE_ANS_MSG:
                agent.current_message_count[QVALUE_ANS_MSG] += 1

                # if a QVALUE_ANS_MSG is requested - we need to give an answer
                should_handle_q_ans = True
                best_data = q_val_ans_dict.get(hash(message.state), None)

                # if this is the first answer for this state - assume it's the best
                if best_data is None:
                    q_val_ans_dict[hash(message.state)] = {BEST_ANS_Q: message.Q_Value,
                                                           BEST_ANS_ID: message.gen_agent_id, BEST_ANS_MSG: message}

                # compare answer to current best ans for message.state
                else:

                    if message.Q_Value < best_data[BEST_ANS_Q] or \
                            (message.Q_Value == best_data[BEST_ANS_Q] and message.gen_agent_id < best_data[BEST_ANS_ID]):
                        best_data[BEST_ANS_Q] = message.Q_Value
                        best_data[BEST_ANS_ID] = message.gen_agent_id
                        best_data[BEST_ANS_MSG] = message

            elif message.type == PUBLIC_STATE_MSG:
                agent.current_message_count[PUBLIC_STATE_MSG] += 1
                agent.create_new_trajectory(message.state, parent_id=message.gen_agent_id,
                                            parent_key=message.trajectory_key)

            elif message.type == BACK_UPDATE_MSG:
                agent.update_backwards(message.state, message.trajectory_key, True)
                break

        # handle QVALUE_ANS_MSG
        if should_handle_q_ans:

            for state_hash in q_val_ans_dict.keys():

                state_best_ans = q_val_ans_dict[state_hash]
                agent_best_action, agent_q = agent.greedy_action(state_best_ans[BEST_ANS_MSG].state)

                # check if requesting agent's q is better than the answers
                if agent_q < state_best_ans[BEST_ANS_Q] or \
                        (agent_q == state_best_ans[BEST_ANS_Q] and agent.id < state_best_ans[BEST_ANS_ID]):
                    state_best_ans[BEST_ANS_Q] = agent_q
                    state_best_ans[BEST_ANS_ID] = agent.id

                # this means the request was for an update and there is no need to involve trajectories
                if state_best_ans[BEST_ANS_MSG].trajectory_key is None:

                    # add the best Q value for state to the agent's dict of states waiting for answer
                    state_reach_prob, _ = agent.states_q_req_dict[state_hash]
                    agent.states_q_req_dict[state_hash] = (state_reach_prob, state_best_ans[BEST_ANS_Q])
                    agent.should_finnish_update = True

                elif state_best_ans[BEST_ANS_ID] != agent.id:

                    # send a message to the best agent for it to advance from this state
                    self.add_message(Message(PUBLIC_STATE_MSG, state_best_ans[BEST_ANS_MSG].state, agent.id,
                                             state_best_ans[BEST_ANS_MSG].trajectory_key, state_best_ans[BEST_ANS_ID]))

                    self.agent_traj_pass[agent.id] += 1

                    # remove the trajectory from forward trajectories of agent
                    agent.remove_forward_trajectory(state_best_ans[BEST_ANS_MSG].trajectory_key)

                    # reset detecting cycles for trajectory
                    agent.traj_states = {}

        # since all messages where read - remove all messages
        self._messages[agent.id] = []

    def add_message(self, message):
        '''
        takes a message object and adds it to the appropriate agents
        :param message: message object
        '''
        if message.type == QVALUE_REQ_MSG:
            for id in self._messages:
                if not id == message.gen_agent_id:
                    self._messages[id].append(message)
        else:
            self._messages[message.rec_agent_id].append(message)

    def plan(self):

        num_states, num_facts = self.get_num_of_states_and_facts()
        self.num_facts = num_facts

        if self.VERBOSE:
            info_str = f'number of states is {num_states}\nnumber of facts is {num_facts}'
            print(colored(info_str, 'cyan'))
            for agent in self._agents:
                print(agent.id, '=', agent.name)

            print(colored('----------------------- planning -----------------------', 'green'))

        self.first_start_time = time.time()
        self.last_start_time = time.time()

        # num of (all possible) actions ~= num of states
        # iterations = len(actions) * 200 * len(self._agents)

        # while ((time.time() - self.first_start_time) / 60) < 15 and not self.reached_convergence(0.7):
        while (self.eval_time / 60) < 25 and not self.reached_convergence(0.7):

            agent = self.select_agent()
            self.process_messages(agent)
            agent.advance_trajectories(self)

        self.save_results()

    def restart_trajectory(self):
        self.current_bad_trajectory_count += 1
        for agent in self.agents:
            agent.remove_open_trajectories()
            agent.should_finnish_update = False
            agent.states_q_req_dict = {}
            agent.prev_seq_count = {}
            agent.prev_state = None
            agent.prev_prev_state = None
            agent.traj_states = {}
            # agent.traj_states = []
            self._messages[agent.id] = []

        # find the best agent to start running after restart
        initial_state = self.agents[0].initial_state
        first_agent = self.get_best_agent(initial_state)

        first_agent.create_new_trajectory(first_agent.initial_state)

    def goal_reached(self, agent_id, trajectory):
        '''
        adds a trajectory to a dict saving trajectories that reached the gaol,
        updates the best_trajectory fields if necessary
        :param agent_id: the id of the agent who's trajectory it is
        :param trajectory: the trajectory
        '''

        goal_end_time = time.time()
        goal_reach_time = goal_end_time - self.last_start_time
        goal_str = 'Found Goal in {0} Seconds'.format(goal_reach_time)

        if self.VERBOSE:
            print(colored(goal_str, 'magenta'))

        self.goal_time_list.append(goal_reach_time)
        self.eval_time += goal_reach_time

        self.bad_trajectory_counts.append(self.current_bad_trajectory_count)
        self.current_bad_trajectory_count = 0

        agent_trajectories = self.goal_trajectories.get(agent_id, [])
        agent_trajectories.append(trajectory)
        self.goal_trajectories[agent_id] = agent_trajectories

        # with one trajectory and no restarts - after reaching goal we need to create
        # a new trajectory from initial state to keep planning
        initial_state = self.agents[0].initial_state
        first_agent = self.get_best_agent(initial_state)

        first_agent.create_new_trajectory(first_agent.initial_state)

        for agent in self.agents:

            # save and reset each agent's current message count every goal reach
            self.agent_messages_count[agent.id].append(agent.current_message_count)
            # print(agent.current_message_count)
            agent.current_message_count = {PUBLIC_STATE_MSG: 0, QVALUE_REQ_MSG: 0, QVALUE_ANS_MSG: 0}

            # save agent expansions
            self.expansion_counts[agent.id].append(agent.current_expansion_count)
            agent.current_expansion_count = 0

            # save Q table for post processing
            self.q_tables[agent.id].append(agent.Q_values)

            # reset visited trajectories for cycle detection
            agent.traj_states = {}

        if len(self.goal_time_list) % 10 == 0:
            self.evaluate_agents()

        self.last_start_time = time.time()

    def evaluate_agents(self):
        '''
        lets each agent run for a set amount of steps and evaluates the cost of each agent
        '''

        actions = 0
        for agent in self.agents:
            actions += len(agent.actions)

        # num of (all possible) actions ~= num of states
        eval_steps = min(actions, 100)

        total_agents_cost = 0
        agent_cost = {}
        goal_reached = False
        steps_taken = 0

        if self.VERBOSE:
            print(colored('----------------------- evaluating agents -----------------------', 'green'))

        # for agent in self.agents:
        #     agent_str = '----------------------- agent {0} {1} Q Table -----------------------'.format(agent.name,
        #                                                                                                agent.type)
        #     print(colored(agent_str, 'green'))
        #     agent.print_Q_Table()

        # print(colored('# eval steps per agent =', 'yellow'), eval_steps)
        # print(colored('# iterations =', 'yellow'), self.eval_iterations)

        for iteration in range(self.eval_iterations):

            # for loop detection
            visited_states = {}

            for agent in self._agents:
                agent_cost[agent.id] = 0

            # this ignores the possibility of different initial states for different agents
            current_state = self.agents[0].initial_state
            agent = self.get_best_agent(current_state)
            eval_actions = []

            no_op_block = False
            action_loop = False

            for step in range(eval_steps):

                # for loop detection
                state_count = visited_states.get(current_state.id, 0)
                state_count += 1
                visited_states[current_state.id] = state_count

                steps_taken += 1

                next_state, cost, has_public_change, action = agent.evaluate(current_state)
                eval_actions.append((action, agent.id))

                if action is not None and action.name == 'no-op':
                    no_op_block = True
                    # print('no-op loop!')
                    break

                if current_state.id in visited_states and visited_states[current_state.id] > 4:
                    action_loop = True
                    # print('EVALUATION IN A LOOP')
                    break

                current_state = next_state
                agent_cost[agent.id] += cost

                # if goal was reached, evaluate returns None
                if current_state is None:
                    goal_reached = True
                    self.best_reached_goal = True
                    break

                if has_public_change and self.USE_PS_RTDP:
                    agent = self.get_best_agent(current_state)

                # in DRTDP we need to get the best agent after every step
                if self.USE_DRTDP:
                    agent = self.get_best_agent(current_state)

                if goal_reached:
                    break

            if no_op_block or action_loop:
                total_agents_cost += eval_steps
            else:
                if self.best_eval_actions is None or len(eval_actions) < len(self.best_eval_actions):
                    self.best_eval_actions = eval_actions

                for agent in self._agents:
                    # print('agent', agent.name, '| eval cost', agent_cost[agent.id])
                    total_agents_cost += agent_cost[agent.id]

            # print(colored('\n# steps taken =', 'yellow'), steps_taken)
            # print(colored('goal reached =', 'yellow'), goal_reached, '\n')
            steps_taken = 0
            goal_reached = False

        self.evaluations.append(total_agents_cost / self.eval_iterations)

        if self.VERBOSE:
            print('total agent cost =', total_agents_cost, 'iterations =', self.eval_iterations, 'res =',
                  (total_agents_cost / self.eval_iterations))
            print(colored('----------------------- evaluations ', 'blue'), len(self.evaluations),
                  colored('-----------------------', 'blue'))
            print(self.evaluations if len(self.evaluations) <= 30 else self.evaluations[-30:], '\n')

    def save_results(self):
        '''
        outputs the results of the planning process
        '''

        total_time_end = time.time()
        if total_time_end % 60 > 1:
            runtime_str = ' $$$$$$$$$$$ Total Planning Run Time is {0} minutes $$$$$$$$$$$'.format(
                (total_time_end - self.first_start_time) / 60)
        else:
            runtime_str = ' $$$$$$$$$$$ Total Planning Run Time is {0} seconds $$$$$$$$$$$'.format(
                total_time_end - self.first_start_time)

        if self.VERBOSE:
            print(colored(runtime_str, 'magenta'))

        file_type = 'png'
        domain_name = self._domains[0].name
        last_hyphen_ind = domain_name.rindex('-')
        domain_name = domain_name[:last_hyphen_ind]

        # -------------- evaluations figure --------------
        if self.VERBOSE:
            print('... saving evaluations figure ...')
        plt.plot(range(len(self.evaluations)), self.evaluations)
        plt.title('Domain - ' + domain_name + ' | eval iterations: ' + str(self.eval_iterations))
        plt.xlabel('evaluation number')
        plt.ylabel('average sum of costs')
        domain_type = domain_name[: domain_name.index('-')]
        file_path = '{0}{1}/{2}/'.format(self.RESULTS_PATH, domain_type, 'evaluations')
        file_name = file_path + domain_name + '.' + file_type

        plt.savefig(file_name, format=file_type, dpi=400)
        plt.cla()

        # -------------- save evaluations as json --------------
        eval_json = json.dumps(self.evaluations)
        f = open(file_path + domain_name + '.json', 'w')
        f.write(eval_json)
        f.close()

        # -------------- save evaluations to overall results csv --------------
        file_path = '{0}{1}/{2}/'.format(self.RESULTS_PATH, domain_type, 'overall')
        file_name = file_path + domain_name + '.csv'
        with open(file_name, 'a') as csvFile:
            writer = csv.writer(csvFile)
            labels_row = ['# goal reach', 'evaluation avg']
            writer.writerow(labels_row)

            for i in range(len(self.evaluations)):
                writer.writerow([i, self.evaluations[i]])
            writer.writerow(['min evaluation', min(self.evaluations)])
        csvFile.close()

        # -------------- goal reach time figure --------------
        if self.VERBOSE:
            print('... saving goal reach time figure ...')
        plt.plot(range(len(self.goal_time_list)), self.goal_time_list)
        plt.title('Domain - ' + domain_name + ' goal reach times')
        plt.xlabel('# goal reach')
        plt.ylabel('time (seconds)')
        file_path = '{0}{1}/{2}/'.format(self.RESULTS_PATH, domain_type, 'goal_reach_time')
        file_name = file_path + domain_name + '.' + file_type

        plt.savefig(file_name, format=file_type, dpi=400)
        plt.cla()

        # -------------- save reach time to overall results csv --------------
        file_path = '{0}{1}/{2}/'.format(self.RESULTS_PATH, domain_type, 'overall')
        file_name = file_path + domain_name + '.csv'

        # for overall json
        total_time = 0
        with open(file_name, 'a') as csvFile:
            writer = csv.writer(csvFile)
            labels_row = ['# goal reach', 'reach time']
            writer.writerow(labels_row)

            for i in range(len(self.goal_time_list)):
                writer.writerow([i, self.goal_time_list[i]])
            total_time = sum(self.goal_time_list)
            total_time_str = '{0} minutes'.format(total_time / 60) if int(total_time / 60) > 0 else \
                '{0} seconds'.format(total_time)
            writer.writerow(['total time', total_time_str])

        csvFile.close()

        # -------------- messages count figure --------------
        if self.VERBOSE:
            print('... saving messages count figure ...')
        domain_type = domain_name[: domain_name.index('-')]
        file_path = '{0}{1}/{2}/'.format(self.RESULTS_PATH, domain_type, 'messages_count')
        message_types = [PUBLIC_STATE_MSG, QVALUE_REQ_MSG, QVALUE_ANS_MSG]
        for agent in self.agents:
            x_axis = list(range(len(self.agent_messages_count[agent.id])))
            file_name = file_path + domain_name + ' Agent - ' + agent.name + '.' + file_type

            color_index = 0
            for message_type in message_types:
                message_axis = list(map(lambda messages_dict: messages_dict[message_type],
                                        self.agent_messages_count[agent.id]))
                plt.plot(x_axis, message_axis, 'C{0}'.format(color_index), label=message_type)
                color_index += 1

            plt.legend()
            plt.xlabel('# times goal was found')
            plt.ylabel('amount of messages per goal find')
            plt.title('Domain - ' + domain_name + ' Agent - ' + agent.name + ' messages count')
            plt.savefig(file_name, format=file_type, dpi=400)
            plt.cla()

        # -------------- save messages to overall results csv --------------
        file_path = '{0}{1}/{2}/'.format(self.RESULTS_PATH, domain_type, 'overall')
        file_name = file_path + domain_name + '.csv'

        # for overall json
        overall_total_messages = 0
        with open(file_name, 'a') as csvFile:
            writer = csv.writer(csvFile)
            agent_names = list(map(lambda agent: agent.name, self.agents))

            labels_row = ['# goal reach'] + agent_names + ['total message count']
            writer.writerow(labels_row)
            for i in range(len(self.goal_time_list)):
                agent_messages = list(map(lambda agent: self.agent_messages_count[agent.id][i][PUBLIC_STATE_MSG] +
                                          self.agent_messages_count[agent.id][i][QVALUE_ANS_MSG] +
                                          self.agent_messages_count[agent.id][i][QVALUE_REQ_MSG], self.agents))
                total_messages = reduce(lambda exp1, exp2: exp1 + exp2, agent_messages)
                writer.writerow([i] + agent_messages + [total_messages])
                overall_total_messages += total_messages
            writer.writerow(['total overall messages', overall_total_messages])
        csvFile.close()

        # -------------- expansion figure --------------
        if self.VERBOSE:
            print('... saving expansions figure ...')
        file_path = '{0}{1}/{2}/'.format(self.RESULTS_PATH, domain_type, 'expansions')
        x_axis = list(range(len(self.goal_time_list)))
        file_name = file_path + domain_name + '.' + file_type
        color_index = 0
        for agent in self.agents:

            agent_axis = self.expansion_counts[agent.id]
            plt.plot(x_axis, agent_axis, 'C{0}'.format(color_index), label=agent.name)
            color_index += 1

        plt.legend()
        plt.xlabel('# times goal was found')
        plt.ylabel('amount of expansions per goal reach')
        plt.title('Domain - ' + domain_name + ' expansions')
        plt.savefig(file_name, format=file_type, dpi=400)
        plt.cla()

        # -------------- save expansion to overall results csv --------------
        file_path = '{0}{1}/{2}/'.format(self.RESULTS_PATH, domain_type, 'overall')
        file_name = file_path + domain_name + '.csv'

        # for overall json
        overall_total_expansions = 0
        with open(file_name, 'a') as csvFile:
            writer = csv.writer(csvFile)
            agent_names = list(map(lambda agent: agent.name, self.agents))
            labels_row = ['# goal reach'] + agent_names + ['total expansions']
            writer.writerow(labels_row)

            for i in range(len(self.goal_time_list)):
                agent_expansions = list(map(lambda agent: self.expansion_counts[agent.id][i], self.agents))
                total_expansions = reduce(lambda exp1, exp2: exp1 + exp2, agent_expansions)
                overall_total_expansions += total_expansions
                writer.writerow([i] + agent_expansions + [total_expansions])
            writer.writerow(['overall total expansions', overall_total_expansions])
        csvFile.close()

        # -------------- bad trajectories figure --------------
        if self.VERBOSE:
            print('------------------- best trajectory -------------------')
            print(self.bad_trajectory_counts)

        total_bad_trajectories = sum(self.bad_trajectory_counts)
        if self.VERBOSE:
            print(colored('total # restarts ' + str(total_bad_trajectories), 'yellow'))

        # -------------- save meta data to overall results csv --------------
        file_path = '{0}{1}/{2}/'.format(self.RESULTS_PATH, domain_type, 'overall')
        file_name = file_path + domain_name + '.csv'
        with open(file_name, 'a') as csvFile:
            writer = csv.writer(csvFile)
            agent_names = list(map(lambda agent: agent.name, self.agents))
            labels_row = ['# actions'] + agent_names + ['total actions']
            writer.writerow(labels_row)

            overall_total_actions = 0
            for agent in self.agents:
                overall_total_actions += len(agent.actions)
                writer.writerow([agent.name, len(agent.actions)])
            writer.writerow(['overall total actions', overall_total_actions])
            num_of_states = self.get_num_of_states_and_facts()
            writer.writerow(["number of states in this problem", num_of_states])
        csvFile.close()

        # for overall json - number of ungrounded actions
        num_actions = 0
        for domain in self._domains:
            num_actions += len(domain.operators)

        # -------------- save meta data to overall results json --------------
        overall_dict = {
            'domain': domain_name,
            'actions': num_actions,
            'facts': self.num_facts,
            'num_agents': len(self.agents),
            'best_cost': min(self.evaluations),
            'messages': overall_total_messages,
            'expansions': overall_total_expansions,
            'trajectories': len(self.goal_time_list),
            'restarts': total_bad_trajectories,
            'planning_time': total_time
        }

        if self.VERBOSE:
            print('... saving overall json ...')
        overall_json = json.dumps(overall_dict)
        alg_str = 'DRTDP' if self.USE_DRTDP else 'PS-RTDP'
        with open(file_path + domain_name + alg_str + '.json', 'w') as f:
            f.write(overall_json)

        # show trajectory passing info
        if self.VERBOSE:
            print('------------------- trajectory passes -------------------')
            for agent in self.agents:
                passes_str = 'agent {0} passed a trajectory {1} times'.format(agent.name, self.agent_traj_pass[agent.id])
                print(colored(passes_str, self.agent_colors[agent.id]))

            # show the best trajectory that happened in evaluation
            print('------------------- best trajectory -------------------')
            print('goal reached = ', self.best_reached_goal)
            # best_eval_actions is of shape - (action, agent.id)
            traj_list = list(map(lambda eval: colored(str(eval[0].name + str(eval[0].param_strings())),
                                                             self.agent_colors[eval[1]]), self.best_eval_actions))
            for action in traj_list:
                print(action)

        pass

    def get_best_agent(self, current_state):
        if current_state is None:
            print('get_best_agent - got None state')
            exit(1)

        best_agent = self.agents[0]
        best_value = float('inf')
        for agent in self.agents:

            best_action_cost = float('inf')
            if current_state in agent.Q_values:
                for action in agent.Q_values[current_state].keys():
                    if agent.Q_values[current_state][action] > 0.0 and \
                            agent.Q_values[current_state][action] < best_action_cost and action.name != 'no-op':
                        best_action_cost = agent.Q_values[current_state][action]

                if best_action_cost == best_value and agent.id < best_agent.id:
                    best_value = best_action_cost
                    best_agent = agent

                if best_action_cost < best_value:
                    best_value = best_action_cost
                    best_agent = agent

            elif current_state.is_initial:

                # let the agent try to advance the initial state (even though the Q values are 0.0)
                # this solves the case where the first agent that runs never makes a public action
                best_value = 0.0
                best_agent = agent

        # if best_value == float('inf'):
        #     print('ALL AGENTS HAD ONLY NO-OP OR ALL ACTIONS ARE UNEXPLORED')

        return best_agent

    def check_q_improvment_on_private(self):
        """
        this function tests whether there are Q_tables where for the same public state but different
        private state, there is a different best action (could be of a different agent)
        the structure of self.q_tables is:
        {agent1.id: [{state: {action: Q_value, action: Q_value, ....}, ...}, ...],
        {agent2.id: [{state: {action: Q_value, action: Q_value, ....}, ...}, ...], ...}
        :return: TBD
        """
        results = {}
        for agent_id in self.q_tables.keys():
            results[agent_id] = []
            table_index = 0
            while table_index < len(self.q_tables[agent_id]):
                table_res = self.check_table(self.q_tables[agent_id][table_index], agent_id)
                results[agent_id].append(table_res)
                table_index += 1

        return results

    def check_table(self, q_table, agent_id):
        found_states = []
        for curr_state in q_table.keys():
            same_public_state_list = list(filter(lambda state: DRTDP.filter_by_same_pub_and_agent_priv_state
                                                                (curr_state, state, agent_id),
                                                 q_table.keys()))

            for other_state in same_public_state_list:
                if other_state is not curr_state:
                    curr_best_action, curr_q_value = self.get_best_action(curr_state, q_table)
                    other_best_action, other_q_value = self.get_best_action(other_state, q_table)

                    if curr_best_action != other_best_action:
                        found_states.append((curr_state, other_state, curr_best_action, other_best_action,
                                             abs(curr_q_value - other_q_value)))

        return found_states

    @staticmethod
    def filter_by_same_pub_and_agent_priv_state(curr_state, other_state, agent_id):
        """
        a filter function for filtering a list of states
        :param curr_state: a state we wish to filter by
        :param other_state: a state that gets filtered
        :param agent_id: the id of the agent by which we should filter
        :return: True - if other_state has the same public literals as curr_state and the same private keys for agent_id
                 False - else
        """

        if curr_state is other_state:
            return False

        if State.has_public_changes(curr_state, other_state):
            return False

        curr_private_keys = curr_state.agent_private_keys[agent_id]
        other_private_keys = other_state.agent_private_keys[agent_id]

        if curr_private_keys != other_private_keys:
            return False

        # if there was no difference in the public literals and no difference in the private keys of the agent
        # there must be a difference in the keys of the other agents - else it's the same state

        # for debug
        if curr_state == other_state:
            print('have the same state twice in q_table')
            print(curr_state.print_state())
            print(other_state.print_state())
            return False

        return True

    def reached_convergence(self, delta):
        """
        checks if the planning process has reached convergence using the evaluations
        :param delta: the threshold that defines satisfactory convergence
        :return: True if convergence has been reached for 3 evaluations
        """
        num_evaluations = len(self.evaluations)
        if num_evaluations < 4:
            return False

        if self.best_eval_actions is not None and \
                (len(self.best_eval_actions) - self.evaluations[num_evaluations - 1]) > delta + 2:
            return False

        diff1 = abs(self.evaluations[num_evaluations - 4] - self.evaluations[num_evaluations - 3])
        diff2 = abs(self.evaluations[num_evaluations - 4] - self.evaluations[num_evaluations - 2])
        diff3 = abs(self.evaluations[num_evaluations - 4] - self.evaluations[num_evaluations - 1])
        diff4 = abs(self.evaluations[num_evaluations - 3] - self.evaluations[num_evaluations - 2])
        diff5 = abs(self.evaluations[num_evaluations - 3] - self.evaluations[num_evaluations - 1])
        diff6 = abs(self.evaluations[num_evaluations - 2] - self.evaluations[num_evaluations - 1])

        actions = 0
        for agent in self.agents:
            actions += len(agent.actions)

        # num of (all possible) actions ~= num of states
        eval_steps = min(actions, 100)

        if (self.evaluations[num_evaluations - 1] < eval_steps) and\
            (self.evaluations[num_evaluations - 2] < eval_steps) and\
            (self.evaluations[num_evaluations - 3] < eval_steps) and \
            (self.evaluations[num_evaluations - 4] < eval_steps):

            if diff1 < delta and diff2 < delta and diff3 < delta and diff4 < delta and diff5 < delta and diff6 < delta:
                return True

        return False

    def get_num_of_states_and_facts(self):
        """
        calculates the number of states in the current problem
        :return: the number of states
        """

        public_predicates = {}
        private_predicates = {}
        sum_of_facts = 0

        for domain in self._domains:

            problem = list(filter(lambda prob: prob.domain == domain.name, self._problems))[0]
            last_index = problem.name.rfind('-')
            agent_name = problem.name[last_index + 1:]

            for predicate in domain.predicates:

                predicate_has_private_arg = False
                arg_types = list(map(lambda arg: Agent.get_all_subtypes(arg.type, domain.type_hierarchy),
                                     predicate.args))
                ordered_objs = []

                # go over each arg type (type of first param, second param ...)
                for possible_types in arg_types:
                    objs_of_subtype = []

                    # for each arg add all the types it can be as a list of types
                    for type in possible_types:
                        if problem.objects.__contains__(type):
                            if problem.objects[type][0].private:
                                predicate_has_private_arg = True
                            objs_of_subtype += problem.objects[type]

                    ordered_objs.append(objs_of_subtype)

                param_combinations = list(itertools.product(*ordered_objs))

                if (predicate.is_private or predicate_has_private_arg) and predicate.name not in private_predicates:
                    private_predicates[predicate.name + '-' + agent_name] = len(param_combinations)
                    sum_of_facts += len(param_combinations)
                elif predicate.name not in public_predicates:
                    public_predicates[predicate.name] = len(param_combinations)
                    sum_of_facts += len(param_combinations)

        total_assignments = list(public_predicates.values()) + list(private_predicates.values())
        num_of_states = reduce(lambda a, b: a * b, total_assignments, 1)
        return num_of_states, sum_of_facts
