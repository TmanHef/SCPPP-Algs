from state import State
from trajectory import Trajectory
from action import Action
from literal import Literal
from predicate import Predicate
from term import Term
from message import Message
from message import PUBLIC_STATE_MSG
from message import BACK_UPDATE_MSG
from message import QVALUE_REQ_MSG
from message import QVALUE_ANS_MSG
from collections import Counter
from termcolor import colored
import itertools
import copy

UNEXPLORED = 'unexplored'
EXPLORED = 'explored'

class Agent(object):

    _id = -1

    def __init__(self, name, type, actions, initial_private_literals, id, planner, domain_name):
        self._name = name
        self._type = type

        # list of the possible actions for the agent
        self._actions = actions
        self.initial_state = None
        self.goal_state = None
        self._id = id
        self._trajectory_key_count = 0
        self._current_trajectory_index = -1
        self.planner = planner
        self.domain = domain_name
        self.should_finnish_update = False
        self.current_state = None
        self.next_state = None
        self.action = None
        self.trajectory = None
        self.local_states = {}
        self.local_unexplored_states = []

        # dict for states that need a Qvalue request answer - {state:{action: [(prob, next_state)]}}
        self.states_q_req_dict = {}

        # answer dict - {state: value}
        self.states_q_ans_dict = {}

        # a list of forward trajectories
        self._forward_trajectories = {}
        self._backward_trajectories = {}
        self._private_literals = {}

        # for cycle detection
        self.prev_state = None
        self.prev_prev_state = None
        self.prev_seq_count = {}
        self.traj_states = {}
        # self.traj_states = []

        # a dict mapping state to a dict of actions and their Qvalue of performing action at state
        self.Q_values = {}

        # initialize the private literals with the literals from the initial state (literals)
        for literal in initial_private_literals:
            self._private_literals[hash(literal)] = literal

        # for monitoring
        self.current_message_count = {PUBLIC_STATE_MSG: 0, QVALUE_REQ_MSG: 0, QVALUE_ANS_MSG: 0}
        self.max_open_traj = 0
        self.current_expansion_count = 0

        if self.planner.VERBOSE:
            print('initialized agent', name, 'with ', len(actions), 'actions')

    @staticmethod
    def build_agent(domain, problem, planner):

        last_index = problem.name.rfind('-')
        agent_name = problem.name[last_index + 1:]
        agent_type = None

        for object_type in problem.objects.keys():
            values_for_type = list(map(lambda term: str(term.value), problem.objects[object_type]))

            if agent_name in values_for_type:
                agent_type = object_type
                break

        actions = Agent.ground_actions(domain.operators, domain.type_hierarchy, problem.objects)
        # actions.append(Action('no-op', [], [], [], 1))

        initial_literals = Literal.literals_from_predicates(problem.init)
        goal_literals = Literal.literals_from_predicates(problem.goal)
        initial_private_literals = list(filter(lambda literal: literal.is_private(), initial_literals))
        initial_public_literals = list(filter(lambda literal: not literal.is_private(), initial_literals))

        Agent._id += 1
        agent = Agent(agent_name, agent_type, actions, initial_private_literals, Agent._id, planner, domain.name)
        initial_private_keys = list(agent._private_literals.keys())
        return agent, initial_public_literals, initial_private_keys, goal_literals

    @staticmethod
    def has_repeating_param(grounded_params):
        '''
        checks if the grounded params have a parameter that appears more than once
        (!) this indicates that actions are not allowed to get two (ore more) parameter values with the same name (!)
        :param grounded_params: a tuple of parameters
        :return: true if there is a parameter that appears more than once in the tuple
        '''

        occurrences_dict = Counter(list(grounded_params))
        occurrences_list = list(occurrences_dict.values())
        return max(occurrences_list) > 1

    @staticmethod
    def ground_actions(operators, type_hierarchy, objects):
        '''
        take the actions and ground them for each object in the problem instance
        :param operators: the operators from the domain
        :param type_hierarchy: the type hierarchy from the parser
        :param objects: all the objects in the problem
        :return: a list of all the grounded action, (!) no actions with the same grounded parameter more than once (!)
        '''
        grounded_actions = []
        for action in operators:

            # ------------ create action possible parameters ------------
            param_types = list(map(lambda param: Agent.get_all_subtypes(param.type, type_hierarchy), action.params))
            ordered_objs = []

            # go over each params type (type of first param, second param ...)
            for possible_types in param_types:
                objs_of_subtype = []

                # for each param add all the types it can be as a list of types
                for type in possible_types:
                    if objects.__contains__(type):
                        objs_of_subtype += objects[type]

                ordered_objs.append(objs_of_subtype)

            param_combinations = list(itertools.product(*ordered_objs))

            # params is the tuple of grounded possible parameters
            for grounded_params in param_combinations:

                # ignore combinations that have the same parameter more than once
                #if not Agent.has_repeating_param(grounded_params):

                # ------------ ground action preconditions to params ------------
                grounded_preconds = Agent.ground_literals_for_action(action.precond, grounded_params, action.params,
                                                                     objects)

                # ------------ ground action effects to params ------------
                grounded_effects = []
                for effect in action.effects:
                    # effect is a Tuple <float, List<Literal>> with the probability and the effects
                    grounded_literals = Agent.ground_literals_for_action(effect[1], grounded_params, action.params,
                                                                         objects)
                    grounded_effect = (effect[0], grounded_literals)
                    grounded_effects.append(grounded_effect)

                # create the grounded action with all the grounded data
                grounded_action = Action(action.name, grounded_params, grounded_preconds, grounded_effects, action.cost)
                grounded_actions.append(grounded_action)

        return grounded_actions

    @staticmethod
    def ground_literals_for_action(literals, grounded_params, action_params, objects):
        grounded_literals = []

        for literal in literals:
            grounded_args = []

            # need to match the predicate args (strings) to the parameters to find the the type
            for arg in literal.predicate.args:
                if not (type(arg) is Term):
                    matching_param = list(filter(lambda param: param.name == arg, action_params))
                    if len(matching_param) == 0:
                        print(
                            'build_agents - precondition parameter name does not match action parameters for action params',
                            action_params)
                        exit(1)

                    # matching_param should only have 1 item since action parameter names should be unique
                    arg_index = action_params.index(matching_param[0])

                    # get the grounded parameter that matches the arg the predicate expects
                    grounded_arg = grounded_params[arg_index]
                    grounded_args.append(Term(value=grounded_arg.value.value, type=grounded_arg.type,
                                              private=grounded_arg.private, name=grounded_arg.value.value))
                else:
                    # if the arg was a constant in the domain file - it will already be grounded to a term here
                    # the privacy of the term has to be determined since it's declared in the problem file
                    if arg.type not in objects:
                        print('constant type', arg.type, 'is missing from problem objects')
                        exit(1)

                    filtered_objs = list(filter(lambda term: term.value.value == arg.value, objects[arg.type]))
                    if len(filtered_objs) == 0:
                        print('constant', arg.value, 'is missing from problem objects')
                        exit(1)

                    # set privacy as defined in the problem obj
                    arg.set_private(filtered_objs[0].private)

                    grounded_args.append(arg)

            # if one of the args is private - the predicate should be private
            # if the predicate is private it's private
            pred_private = literal.predicate.is_private
            for term in grounded_args:
                if term.private:
                    pred_private = True

            # create a new literal with the grounded predicate args
            grounded_literal = Literal(Predicate(literal.predicate.name, pred_private, grounded_args),
                                       literal.is_positive())
            grounded_literals.append(grounded_literal)

        return grounded_literals

    @staticmethod
    def get_all_subtypes(type, types_hierarchy):

        # in case this type has no subtypes
        if type not in types_hierarchy:
            return [type]

        acc_types = [type]
        for sub_type in types_hierarchy[type]:
            acc_types = acc_types + Agent.get_all_subtypes(sub_type, types_hierarchy)

        return acc_types

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    @property
    def actions(self):
        return self._actions

    @property
    def private_literals(self):
        return self._private_literals

    def expand_states(self):

        # if everything is explored return and wait for messages or VI stage
        if len(self.local_unexplored_states) == 0:
            return

        current_state = self.local_unexplored_states.pop(0)

        if current_state not in self.local_states:

            # first time seeing this state
            self.local_states[current_state] = UNEXPLORED

        curr_actions = current_state.possible_actions(self.id, self.type, self.name, self.actions, self._private_literals)
        for action in curr_actions:
            for effect in action.effects:
                next_state = self.get_effected_state(current_state, effect)
                if next_state not in self.local_states:
                    self.local_unexplored_states.append(next_state)
                    self.local_states[next_state] = UNEXPLORED

                if State.has_public_changes(current_state, next_state):

                    # update other agents of new public state so they can expand it as well
                    self.planner.add_message(Message(PUBLIC_STATE_MSG, next_state, self.id, None,))

        # after going over all actions and effects - current state is explored
        self.local_states[current_state] = EXPLORED

    def init_q_table(self):
        for state in self.local_states.keys():
            curr_actions = state.possible_actions(self.id, self.type, self.name)

            action_q_values = {}
            for action in curr_actions:

                # init Q values with 0.0
                action_q_values[action] = 0.0

            self.Q_values[state] = action_q_values

    def iteration_update(self):
        if self.should_finnish_update:
            self.finnish_update()
        else:
            for state in self.local_states.keys():
                actions = state.possible_actions(self.id, self.type, self.name)
                self.states_q_req_dict[state] = {}
                for action in actions:
                    self.states_q_req_dict[state][action] = []
                    self.request_Qvalues(action, state)

    def greedy_action(self, state):
        '''returns the best action by greedy policy and it's cost
           as a tuple (action, cost)'''

        # add the state and it's possible actions to Q_values if they do not exist
        if state not in self.Q_values:
            return None, float('inf')

        actions = self.Q_values[state].keys()
        if len(actions) == 0:
            return None, float('inf')

        greedy = None
        best_val = float('inf')

        for action in actions:
            act_val = self.Q_values[state][action]
            if self.is_action_better(greedy, best_val, action, act_val, state):
                greedy = action
                best_val = act_val

        if greedy is None:

            # there is no best action yet, choose some action - todo - think about choosing the first action
            #  always (or heuristic)
            action_ind = self.planner.random.randrange(0, len(actions), 1)
            print('in greedy action using random =', action_ind)
            greedy = actions[action_ind]

        return greedy, best_val

    def greedy_action_evaluation(self, state):
        '''returns the best action by greedy policy and it's cost
           as a tuple (action, cost) - ignoring unexplored actions (actions with Q value of 0.0)'''

        # add the state and it's possible actions to Q_values if they do not exist
        if state not in self.Q_values:
            return None, float('inf')

        actions = self.Q_values[state].keys()
        if len(actions) == 0:
            return None, float('inf')

        greedy = None
        best_val = float('inf')

        for action in actions:
            act_val = self.Q_values[state][action]
            if act_val > 0.0 and action.name != 'no-op' and self.is_action_better(greedy, best_val, action, act_val, state):
                greedy = action
                best_val = act_val

        if greedy is None:
            for action in actions:
                greedy = action
                break

        return greedy, best_val

    def Qvalue(self, action, state):
        '''
        :param action: a grounded action
        :param state: the state at which the action is performed
        :param next_state: the state reached
        :return: the Q value of the grounded action at state (cost(action, state) + SUM( Pa(s'|s) * s'.value)
        '''

        action_cost = action.cost

        # go over possible outcomes and calculate the sum
        expected_costs_sum = 0.0

        # for the "stay in place" case check the probabilities sum
        prob_sum = 0.0

        # effect = Tuple<probability, List<Literals>>
        for effect in action.effects:
            next_state = self.get_effected_state(state, effect)
            if next_state == state:
                continue

            next_best_action, _ = self.greedy_action(next_state)
            prob_sum += effect[0]
            # the expected cost of the next state is the Q value of it's best action
            expected_costs_sum += effect[0] * self.Q_values[next_state][next_best_action]

        # there is an implicit possible effect of staying in place
        if prob_sum < 1.0:

            # add the cost of staying in place
            expected_costs_sum += (1.0 - prob_sum) * self.Q_values[state][action]

        action_cost += expected_costs_sum
        return action_cost

    def finnish_update(self):
        iteration_q_table = {}
        for state in self.states_q_req_dict.keys():
            for action in self.states_q_req_dict[state].keys():
                action_cost = action.cost

                # go over possible outcomes and calculate the sum
                expected_costs_sum = 0.0

                # list of Tuple(reach prob, next state)
                for reach_prob, next_state in self.states_q_req_dict[state][action]:
                    next_state_q = 0 if self.is_goal(next_state) else self.states_q_ans_dict[next_state]
                    expected_costs_sum += reach_prob * next_state_q

                action_cost += expected_costs_sum

                if state not in iteration_q_table:
                    iteration_q_table[state] = {}

                iteration_q_table[state][action] = action_cost

        self.Q_values = iteration_q_table
        self.states_q_req_dict = {}
        self.states_q_ans_dict = {}
        self.should_finnish_update = False

    def request_Qvalues(self, action, state):
        '''
        :param action: a grounded action
        :param state: the state at which the action is performed
        :param next_state: the state reached
        :return: the Q value of the grounded action at state (cost(action, state) + SUM( Pa(s'|s) * s'.value)
        '''

        # for the "stay in place" case check the probabilities sum
        prob_sum = 0.0

        # effect = Tuple<probability, List<Literals>>
        for effect in action.effects:
            next_state = self.get_effected_state(state, effect)

            # if the effect of some action results in the same state -
            # this will be handled by the "staying in place" case
            # NOTE - this happened for example in logistics with a plane flying from apt1 to apt1
            # clearly not a valid action but the grounding mechanism can be improved
            if next_state == state:
                continue

            if not State.has_public_changes(state, next_state):
                _, own_q = self.greedy_action(next_state)
                self.states_q_ans_dict[next_state] = own_q
            else:
                self.planner.add_message(Message(QVALUE_REQ_MSG, next_state, self.id, None))

            prob_sum += effect[0]
            self.states_q_req_dict[state][action].append((effect[0], next_state))

        # there is an implicit possible effect of staying in place
        if prob_sum < 1.0:

            # add the cost of staying in place
            # self.planner.add_message(Message(QVALUE_REQ_MSG, state, self.id, None))
            self.states_q_req_dict[state][action].append((1.0 - prob_sum, state))
            _, own_q = self.greedy_action(state)
            self.states_q_ans_dict[state] = own_q

    # simulate stochastic effect
    # todo - think about this
    def get_next_state(self, action, state):

        # sort effects by probability in ascending order
        action.effects.sort(key=lambda effect: effect[0])
        # print('sorted effects:', action.effects)

        selected_effect = None
        rand = self.planner.random.random()
        # print('random =', rand)

        acc_prob = 0.0

        for effect in action.effects:
            acc_prob += effect[0]
            if rand < acc_prob:
                selected_effect = effect
                break

        if selected_effect is None:

            # by default if there is no effect - the state stays the same
            return state

        next_state = self.get_effected_state(state, selected_effect)
        return next_state

    def get_effected_state(self, state, effect):
        '''
        use this function to get a state object (create one if it doesn't already exists) that results from
        performing effect on state
        :param state: a state object
        :param effect: an effect tuple <effect prob, literals list>
        :return: the effected state object
        '''

        literals, private_keys = self.effect_state(state, effect)
        effected_agent_keys = copy.deepcopy(state.agent_private_keys)
        effected_agent_keys[self.id] = private_keys
        effected_state = self.get_state_for_literals(literals, effected_agent_keys)
        return effected_state

    def effect_literals(self, current_literals, effect_literals):
        '''
        takes literals from effect and uses them to change the state
        :param current_literals: list of literals of the current state
        :param effect_literals: list of literals that change the state
        :return: list of literals after performing the effect, returns only the positive literals
        '''

        effect_preds = list(map(lambda literal: str(literal.predicate), effect_literals))

        # if a literal exists in the current state, the effect will have the wanted positivity value
        # take the literals missing from the effect and add the effect literals to them
        missing_literals = list(filter(lambda literal: not str(literal.predicate) in effect_preds, current_literals))
        all_literals = missing_literals + effect_literals

        # filter out the negative literals - if a predicate does not appear it is assumed to be false
        positive_literals = list(filter(lambda literal: literal.is_positive(), all_literals))

        return positive_literals

    # todo - think if this should be a utility
    def effect_state(self, state, effect):
        '''
        takes literals from effect and uses them to change the state
        :param state: a state
        :param effect: a tuple (effect_prob, list<Literal>)
        :return: a list of the public literals with the effect changes, a list of private keys with the effect changes
        '''

        # effect is a tuple of (effect_prob, list<Literal>)
        # get the effected private literals
        current_private_literals = [self._private_literals[x] for x in state.agent_private_keys[self.id]]
        effect_private_literals = list(filter(lambda literal: literal.is_private(), effect[1]))
        effected_private_literals = self.effect_literals(current_private_literals, effect_private_literals)

        # get the effected public literals
        current_public_literals = state.literals
        effect_public_literals = list(filter(lambda literal: not literal.is_private(), effect[1]))
        effected_public_literals = self.effect_literals(current_public_literals, effect_public_literals)

        # get the keys of the private literals and remove the private literals from the list
        private_keys = []
        for literal in effected_private_literals:
            key = hash(literal)
            val = self._private_literals.get(key, None)

            # if the literal wasn't in private_literals, add it
            if val is None:
                self._private_literals[key] = literal

            private_keys.append(key)

        return effected_public_literals, private_keys

    def get_state_for_literals(self, literals, agent_private_keys):
        '''returns the state corresponding to the literals
           checks in private and public states is state was created
           if not, creates the state and saves it
           assumes that public states have no private literals'''

        literals_hash = State.get_hash(literals, agent_private_keys)

        new_state = self.planner.states.get(literals_hash)
        if new_state is not None:
            return new_state

        # the state was not created yet
        new_state = State(literals, agent_private_keys, self.planner)
        new_state.id = len(self.planner.states)
        self.planner.states[hash(new_state)] = new_state

        # if len(self.planner.states) > 8000:
        #     # for debug
        #     agent_color = self.planner.agent_colors[self.id]
        #     print(colored('created new state!', agent_color))
        #     print(colored(new_state.print_full_state(), agent_color))

        if not hash(new_state) == literals_hash:
            print('-------------created new state with different hash-------------')
            print(literals)
            print(agent_private_keys)

        if len(self.planner.states) % 100 == 0 and self.planner.VERBOSE:
            print(len(self.planner.states), 'states added')

        # if len(self.planner.states) >= 10000:
        #     print('created new state')
        #     print(new_state.print_state())

        return new_state

    def is_goal(self, state):
        '''checks if state contains all the goal state's literals
           assumes goal states are public'''

        return state.contains_public_state(self.goal_state)

    def evaluate(self, current_state):
        '''
        lets the agent run from current_state until public change or goal reached
        :param current_state: the current_state
        :return: the current state reached, the cost accumulated so far
        '''

        agent_color = self.planner.agent_colors[self.id]
        action, action_cost = self.greedy_action_evaluation(current_state)

        if action is not None:

            next_state = self.get_next_state(action, current_state)
            if len(self.planner.evaluations) > 20 and self.planner.VERBOSE:
                print(colored('agent', agent_color), self.name, colored('doing --->', agent_color), action.name,
                      action.param_strings(), colored('at cost:', agent_color), action_cost,
                      colored('at state:', agent_color), current_state.id, colored('resulting with state:', agent_color),
                      next_state.id)

            if self.is_goal(next_state):
                return None, action.cost, True, action

            if State.has_public_changes(current_state, next_state):
                return next_state, action.cost, True, action

            return next_state, action.cost, False, action
        else:
            print('no possible actions for agent', self.name)
            return current_state, 0, False, None

    def is_action_better(self, first_action, first_val, second_action, second_val, state):
        """
        checks if an action is better than the other at a specified state.
        may use heuristics depending on the domain
        :param first_action: the first action
        :param first_val: the Q_value for that action
        :param second_action: the second action
        :param second_val: the Q_value for the second action
        :param state: the state at which the actions are performed
        :return: True if second_action is better than first_action
        """

        if first_action is None:
            return True

        if second_val < first_val:
            return True

        if not self.planner.use_heuristic:
            # if no heuristic should be used, return false since both actions have the same value
            return False

        # only blocks world supports heuristics for now
        # if 'blocks' in self.domain:

        # use heuristic to break ties
        if second_val == first_val:
            first_act_goal_pred_count = 0
            second_act_goal_pred_count = 0

            for effect in first_action.effects:
                effected_state = self.get_effected_state(state, effect)
                first_act_goal_pred_count += effect[0] * effected_state.get_common_literals(self.goal_state)

            for effect in second_action.effects:
                effected_state = self.get_effected_state(state, effect)
                second_act_goal_pred_count += effect[0] * effected_state.get_common_literals(self.goal_state)

            # if an action has effects with more literals in common with goal - it's better
            if second_act_goal_pred_count > first_act_goal_pred_count:
                # print('heuristic found a better action!')
                # print('action 1 has common literals expectancy of', first_act_goal_pred_count)
                # print('action 2 has common literals expectancy of', second_act_goal_pred_count)

                # add noise to the heuristic - this means with some probability the heuristic does not return
                # the best action but the other action
                # if self.planner.random.random() >= self.planner.heuristic_noise:
                #     print('in heuristic using random')
                #     return True
                # else:
                #     return False
                return True

            return False
            # return False
        return False
