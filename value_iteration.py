from message import PUBLIC_STATE_MSG
from message import BACK_UPDATE_MSG
from message import QVALUE_REQ_MSG
from message import QVALUE_ANS_MSG
from message import Message
from vi_agent import Agent, UNEXPLORED, EXPLORED
from state import State
from termcolor import colored
from functools import reduce
import matplotlib.pyplot as plt
import time
import random
import json
import csv
import itertools


class ValueIteration(object):

    RESULTS_PATH = './result_graphs/'

    def __init__(self, domains, problems, iterations, eval_iterations):
        self._domains = domains
        self._problems = problems

        # a dictionary mapping public states by their hash (public states have no private literals)
        self.states = {}
        self._current_agent = -1
        self.iterations = iterations
        self.eval_iterations = eval_iterations
        self.current_iteration = 0
        self.all_agents_done = False
        self.use_heuristic = False

        # for monitoring
        self.goal_trajectories = {}
        self.best_goal_trajectory = None
        self.best_goal_trajectory_cost = float('inf')
        self.iterations_time_list = []
        self.evaluations = []
        self.agent_colors = ['red', 'green', 'blue',  'cyan', 'white', 'yellow', 'grey', 'magenta']
        self.last_start_time = None
        self.first_start_time = None
        self.eval_time = 0
        self.best_eval_actions = None
        self.best_reached_goal = False
        self.bad_trajectory_counts = []
        self.current_bad_trajectory_count = 0
        self.VERBOSE = False
        self.num_facts = 0

        # always use the same seed for easier debugging
        self.random = random
        self.random.seed(100)

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
            agent.local_unexplored_states.append(initial_state)

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
                best_data = q_val_ans_dict.get(message.state, None)

                # if this is the first answer for this state - assume it's the best
                if best_data is None:
                    q_val_ans_dict[message.state] = {BEST_ANS_Q: message.Q_Value,
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
                if message.state not in agent.local_states:
                    agent.local_unexplored_states.append(message.state)

            elif message.type == BACK_UPDATE_MSG:
                agent.update_backwards(message.state, message.trajectory_key, True)
                break

        # handle QVALUE_ANS_MSG
        if should_handle_q_ans:

            for state in q_val_ans_dict.keys():

                state_best_ans = q_val_ans_dict[state]
                agent_best_action, agent_q = agent.greedy_action(state_best_ans[BEST_ANS_MSG].state)

                # check if requesting agent's q is better than the answers
                if agent_q < state_best_ans[BEST_ANS_Q] or \
                        (agent_q == state_best_ans[BEST_ANS_Q] and agent.id < state_best_ans[BEST_ANS_ID]):
                    state_best_ans[BEST_ANS_Q] = agent_q
                    state_best_ans[BEST_ANS_ID] = agent.id

                # add the best Q value for state to the agent's dict of states waiting for answer
                agent.states_q_ans_dict[state] = state_best_ans[BEST_ANS_Q]
                agent.should_finnish_update = True

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

            # if rec agent not specified, send to all
            if message.rec_agent_id is None:
                for agent in self.agents:
                    self._messages[agent.id].append(message)
            else:
                self._messages[message.rec_agent_id].append(message)

    def plan(self):

        if self.VERBOSE:
            for agent in self._agents:
                print(agent.id, '=', agent.name)

            print(colored('----------------------- planning -----------------------', 'green'))

        self.first_start_time = time.time()
        self.last_start_time = time.time()

        # keep trying to expand state util all agents are done
        while (time.time() - self.first_start_time) / 60 < 15:

            agent = self.select_agent()
            self.process_messages(agent)

            # if after processing messages, no agent has unexplored states then we are done
            still_expanding = False
            for agent in self.agents:
                if not len(agent.local_unexplored_states) == 0:
                    still_expanding = True
                    break

            if not still_expanding:
                break

            agent.expand_states()

        if self.VERBOSE:
            print('done')
            for agent in self.agents:
                if UNEXPLORED in agent.local_states.values():
                    print(f'agent {agent.name} has unexplored states')
                print(f'agent {agent.name} explored {len(agent.local_states.keys())} states')

        # after all states are expanded - perform multi agent value iteration
        for agent in self.agents:
            agent.init_q_table()
            if self.VERBOSE:
                print(f'agent {agent.name} initialized Q table')

        while not self.reached_convergence(0.7) and (sum(self.iterations_time_list)) / 60 < 15:
            for agent in self.agents:
                self.process_messages(agent)
                agent.iteration_update()

            self.current_iteration += 1

            # it takes 2 passes over all agents to complete one update iteration
            if self.current_iteration % 20 == 0:
                self.iterations_time_list.append(time.time() - self.last_start_time)
                self.evaluate_agents()
                self.last_start_time = time.time()

        self.save_results()

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

                if has_public_change:
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

            if self.VERBOSE:
                print(colored('\n# steps taken =', 'yellow'), steps_taken)
                print(colored('goal reached =', 'yellow'), goal_reached, '\n')
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
        if float(total_time_end - self.first_start_time) / 60 > 1:
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
        domain_type = domain_name[: domain_name.index('-')]
        file_path = '{0}{1}/{2}/'.format(self.RESULTS_PATH, domain_type, 'evaluations')

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
        plt.plot(range(len(self.iterations_time_list)), self.iterations_time_list)
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
        with open(file_name, 'a') as csvFile:
            writer = csv.writer(csvFile)
            labels_row = ['# goal reach', 'reach time']
            writer.writerow(labels_row)

            for i in range(len(self.iterations_time_list)):
                writer.writerow([i, self.iterations_time_list[i]])
            total_time = sum(self.iterations_time_list)
            total_time_str = '{0} minutes'.format(total_time / 60) if int(total_time / 60) > 0 else \
                '{0} seconds'.format(total_time)
            writer.writerow(['total time', total_time_str])

        csvFile.close()

        # for overall json
        overall_total_messages = 0
        for agent in self.agents:
            agent_messages = sum(agent.current_message_count.values())
            overall_total_messages += agent_messages

        # for overall json
        overall_total_expansions = sum([len(agent.local_states.keys()) for agent in self.agents])

        # for overall json - number of ungrounded actions
        num_actions = 0
        for domain in self._domains:
            num_actions += len(domain.operators)

        # -------------- save meta data to overall results json --------------
        if self.reached_convergence(0.7):
            overall_dict = {
                'domain': domain_name,
                'actions': num_actions,
                'facts': self.get_num_of_states_and_facts()[1],
                'num_agents': len(self.agents),
                'best_cost': min(self.evaluations),
                'messages': overall_total_messages,
                'expansions': overall_total_expansions,

                # a full iteration takes a round of value requests and a round of updates
                # so current_iteration is increased twice
                'iterations': self.current_iteration / 2,
                'planning_time': sum(self.iterations_time_list)
            }
        else:
            overall_dict = {}

        if self.VERBOSE:
            print('... saving overall json ...')
        overall_json = json.dumps(overall_dict)
        alg_str = 'VI'
        with open(file_path + domain_name + alg_str + '.json', 'w') as f:
            f.write(overall_json)

        # show trajectory passing info
        if self.VERBOSE:

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

            elif current_state.is_initial and \
                    len(current_state.possible_actions(agent.id, agent.type, agent.name)) > 0:

                # let the agent try to advance the initial state (even though the Q values are 0.0)
                # this solves the case where the first agent that runs never makes a public action
                best_value = 0.0
                best_agent = agent

        # if best_value == float('inf'):
        #     print('ALL AGENTS HAD ONLY NO-OP OR ALL ACTIONS ARE UNEXPLORED')

        return best_agent

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
