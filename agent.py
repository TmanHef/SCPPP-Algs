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

        # dict for states that need a Qvalue request answer
        self.states_q_req_dict = {}

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

        # add explicit "other agents" action
        self.other_agent_action = Action('other agent', [], [], [(1.0, [])], 0.5)

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
        actions.append(Action('no-op', [], [], [], 1))

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

    def advance_trajectories(self, planner):

        # try to advance until it's time to give up run time
        while True:

            if self.planner.USE_DRTDP:
                if self.should_finnish_update:

                    # returns if we should break from the main loop
                    if self.resume_previous():
                        break

            traj = self.select_trajectory()

            # if the agent has no open trajectories, skip his turn
            if traj is None:
                return

            current_state = traj.current_state()

            if self.planner.USE_PS_RTDP:

                # for cycle detection
                state_count = self.traj_states.get(current_state.id, 0)
                state_count += 1
                self.traj_states[current_state.id] = state_count

            self.current_expansion_count += 1

            # get the best action in a greedy fashion for the current state in the trajectory
            action, action_cost = self.greedy_action(current_state)

            if action is not None:

                # for debug
                agent_color = self.planner.agent_colors[self.id]
                debug_str = 'agent {0} doing action: {1} in state {2}'.format(self.name, str(action.name), current_state.id)
                # print(colored(debug_str, agent_color))

                # stochastically get the state reached by performing action at state
                next_state = self.get_next_state(action, current_state)

                if self.planner.USE_DRTDP:

                    # save data to use after agents answer
                    self.current_state = current_state
                    self.next_state = next_state
                    self.action = action
                    self.trajectory = traj

                    self.request_Qvalues(action, current_state)

                    # break from loop to get answers from other agents
                    break

                # the agent performing the action needs to do the update
                self.update(current_state, next_state, action, traj.trajectory_key)

                # for cycle detection
                if next_state.id in self.traj_states.keys() and self.traj_states[next_state.id] > 4:
                    self.planner.restart_trajectory()
                    break

                if self.is_goal(next_state):
                    # next_state is the goal state - add traj to it's agent_trajectories here
                    # since this trajectory will be removed and the state will not get updated with update function
                    next_state.add_agent_trajectory(self.id, traj.trajectory_key)
                    trajectory = self._forward_trajectories.pop(traj.trajectory_key)
                    self._backward_trajectories[traj.trajectory_key] = trajectory

                    planner.goal_reached(self.id, traj)

                    if len(self.planner.agents) == 1:
                        print('NOT SUPPORTING ONE AGENT')
                        exit(1)
                        self.create_new_trajectory(self.initial_state)

                    # break from main loop
                    break

                # if we have more than 1 agent, no matter what the next state is,
                # request Q values to choose the best agent to act
                # we do this instead of requesting on public changes to solve the issue of agents that can
                # perform private actions and never reach a public state change
                if len(self.planner.agents) > 1:

                    # send message only on public changes
                    if State.has_public_changes(current_state, next_state):

                        # request best Q Value of next state from other agents
                        self.planner.add_message(Message(QVALUE_REQ_MSG, next_state, self.id, traj.trajectory_key,
                                                         prev_state=current_state, prev_action=action))

                        # next thing to do is get all the answers, so now we break
                        break

                    # break to allow restarts even if no public change was made
                    break

            # no possible action exists from the current state of the trajectory - we can remove it
            else:
                if len(self.planner.agents) > 1:

                    # request best Q Value of next state from other agents
                    self.planner.add_message(Message(QVALUE_REQ_MSG, current_state, self.id, traj.trajectory_key))
                else:
                    print('one agent running and has nothing to do')
                    exit(1)
                break

    def create_new_trajectory(self, first_state, parent_id=None, parent_key=None):
        '''
        create a new trajectory starting from first_state and adds it to forward_trajectories
        :param first_state: the state from which the agent starts advancing the trajectory
        :param parent_id: if a parent exists -> his id, else -> None
        :param parent_key: if a parent exists -> the key to the trajectory that resulted in this one, else -> None
        '''

        # for now - limit the number of trajectories to 10
        # if len(self._forward_trajectories.keys()) < 5:

        if not self.trajectory_exists(first_state):
            self._trajectory_key_count += 1
            new_traj = Trajectory(first_state, self._id, self._trajectory_key_count, self.planner, parent_id, parent_key)
            self._forward_trajectories[self._trajectory_key_count] = new_traj

            # for monitoring
            # traj_str = 'agent {0} created new trajectory starting at {1}and now has {2} open trajectories'.format(
            #     self.name, first_state.print_state(), len(self._forward_trajectories))
            # print(colored(traj_str, self.planner.agent_colors[self.id]))

            if len(self._forward_trajectories.keys()) > self.max_open_traj:
                self.max_open_traj = len(self._forward_trajectories.keys())
                # traj_count_str = 'agent {0} updated max traj count to {1}'.format(self.name, self.max_open_traj)
                # print(colored(traj_count_str, self.planner.agent_colors[self.id]))

    def remove_forward_trajectory(self, trajectory_key):
        traj = self._forward_trajectories.pop(trajectory_key)
        self._backward_trajectories[trajectory_key] = traj

    def remove_open_trajectories(self):
        self._forward_trajectories.clear()

    # todo
    def update_backwards(self, state, traj_key, from_message):
        '''
        goes backwards from state (or from last state of traj if state is None),
        recalculates the cost of the action performed at each state and if needed updates the total expected
        cost of the state, sends messages to other agents passing through the state
        :param state: a state object from which to start the backwards pass
        :param traj_key: the key of the trajectory which this state belongs to
        '''

        # ------------ get the relevant trajectory ------------
        if traj_key in self._forward_trajectories:
            traj = self._forward_trajectories[traj_key]
        elif traj_key in self._backward_trajectories:
            traj = self._backward_trajectories[traj_key]
        else:
            print('trajectory key', traj_key, 'not found for backward update by agent', self.id)
            return

        # create a copy of the path where the current state is the state given
        # use shallow copy to work with the same state objects without breaking the path list
        path_copy = copy.copy(traj.path)

        print('agent', self.name, 'backwards update path of length', len(path_copy))

        if state is not None:
            # go from end to beginning of the list and remove state until state is found
            while len(path_copy) > 0 and not path_copy[len(path_copy) - 1][0] == state:
                path_copy.pop(len(path_copy) - 1)

            if len(path_copy) == 0:
                print('backwards_update ----- state was not found in trajectory')
                exit(1)

        cost_not_changed = False

        # go backwards over the path and update the expected cost of the actions
        # and the cost of the state if needed - send messages to relevant agents if state cost changed
        for traj_state, action in reversed(path_copy):

            # if the state is the goal - it was just updated in forward pass
            if not self.is_goal(traj_state):

                if action is None:
                    action, _ = self.greedy_action(traj_state)

                # calculate the Qvalue of the action - it might change since the next state reached by the action
                # might have had it's cost updated
                new_cost = self.Qvalue(action, traj_state)
                traj_state.update_action_cost(action, new_cost, self.id)
                # print('agent', self.name, 'back updated state')

                # for debug
                if traj_state is self.initial_state:
                    print(self.name, 'back updating initial state')

                # if the cost is better than the cost of the state - update it and send messages to other agents
                if new_cost != traj_state.cost:
                    traj_state.cost = new_cost

                    if traj_state is self.initial_state:
                        break

                    # ------------ send messages to passing agents ------------
                    # agent_ids_and_trajectories = traj_state.get_passing_agents_ids(exclude=self.id)
                    # for agent_id in agent_ids_and_trajectories:
                    #
                    #     # agent_id is (id, list<trajectory keys>)
                    #     for rec_agent_traj_key in agent_id[1]:
                    #         message = Message(BACK_UPDATE_MSG, traj_state, self.id, rec_agent_traj_key,
                    #                           rec_agent_id=agent_id[0])
                    #         print('agent', self.id, 'sending back update msg to agent', agent_id[0])
                    #         self.planner.add_message(message)

                # the cost wasn't improved - nothing to update further
                else:
                    cost_not_changed = True
                    break

        # if cost change was propagated through the entire path and the trajectory has a parent trajectory -
        # send a message
        if not cost_not_changed and traj.parent_agent_id is not None:
            message = Message(BACK_UPDATE_MSG, None, self.id, traj.parent_trajectory_key,
                              rec_agent_id=traj.parent_agent_id)
            self.planner.add_message(message)
            print(self.id, '(', traj_key, ')', '---update backward--->', traj.parent_agent_id, '(', traj.parent_trajectory_key, ')')

        if not from_message:
            trajectory = self._forward_trajectories.pop(traj_key)
            self._backward_trajectories[traj_key] = trajectory

        print(self.name, 'finished updating backwards')
        pass

    # todo - for now - round robin selection
    def select_trajectory(self):
        ''' returns a trajectory'''

        # if there are no trajectories, return None
        if len(self._forward_trajectories.keys()) == 0:
            return None

        next_traj_ind = (self._current_trajectory_index + 1) % len(self._forward_trajectories.keys())
        self._current_trajectory_index = next_traj_ind
        next_traj_key = list(self._forward_trajectories.keys())[next_traj_ind]
        return self._forward_trajectories[next_traj_key]

    def greedy_action(self, state):
        '''returns the best action by greedy policy and it's cost
           as a tuple (action, cost)'''

        # add the state and it's possible actions to Q_values if they do not exist
        if state not in self.Q_values:
            curr_actions = state.possible_actions(self.id, self.type, self.name, self.actions, self._private_literals)

            action_q_values = {}
            for action in curr_actions:
                # init Q values with 0.0
                action_q_values[action] = 0.0

            self.Q_values[state] = action_q_values

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
            # print('NEW STATE REACHED IN EVALUATION')
            curr_actions = state.possible_actions(self.id, self.type, self.name, self.actions, self._private_literals)

            action_q_values = {}
            for action in curr_actions:
                # init Q values with 0.0
                action_q_values[action] = 0.0

            self.Q_values[state] = action_q_values

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

    def finnish_update(self, current_state, next_state, action, trajectory_key):
        action_cost = action.cost

        # go over possible outcomes and calculate the sum
        expected_costs_sum = 0.0

        # effect = Tuple<probability, List<Literals>>
        for reach_prob, q_val in self.states_q_req_dict.values():

            # the expected cost of the next state is the Q value of it's best action (including 'other agent' action)
            expected_costs_sum += reach_prob * q_val

        action_cost += expected_costs_sum

        self.Q_values[current_state][action] = action_cost

        self.states_q_req_dict = {}
        self.should_finnish_update = False

        curr_traj = self._forward_trajectories[trajectory_key]

        # ---------- update the current trajectory ----------
        # set the action performed in current state
        curr_traj.set_current_action(action)

        # add the new state to the trajectory with None action since we don't know yet what action will be performed
        curr_traj.add_state_action(next_state, None)

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

            self.planner.add_message(Message(QVALUE_REQ_MSG, next_state, self.id, None))
            prob_sum += effect[0]
            self.states_q_req_dict[hash(next_state)] = (effect[0], None)

        # there is an implicit possible effect of staying in place
        if prob_sum < 1.0:

            # add the cost of staying in place
            self.planner.add_message(Message(QVALUE_REQ_MSG, state, self.id, None))
            self.states_q_req_dict[hash(state)] = (1.0 - prob_sum, None)

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

    def update(self, current_state, next_state, action, traj_key):
        '''
        for single agent
        update the trajectory of the agent with the new state and action
        update the expected total cost of the action at current_state
        :param current_state: the current state
        :param next_state: the next state
        :param action: the action used to transition between the states
        :param traj_key: the current trajectory that is advancing
        '''

        # calculate the Qvalue of the chosen action and update Q table
        action_cost = self.Qvalue(action, current_state)

        self.Q_values[current_state][action] = action_cost

        curr_traj = self._forward_trajectories[traj_key]

        # ---------- update the current trajectory ----------
        # set the action performed in current state
        curr_traj.set_current_action(action)

        # add the new state to the trajectory with None action since we don't know yet what action will be performed
        curr_traj.add_state_action(next_state, None)

        # for cycle detection
        self.prev_prev_state = self.prev_state
        self.prev_state = current_state

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

    def trajectory_exists(self, first_state):
        '''
        checks if there is a trajectory that starts with first_state
        :param first_state: the first state of a trajectory
        :return: True if a trajectory exists else False
        '''

        for traj in self._forward_trajectories.values():
            traj_first_state = traj.path[0][0]
            if first_state == traj_first_state:
                return True

        return False

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
            # if len(self.planner.evaluations) > 80:
            #     print(colored('agent', agent_color), self.name, colored('doing --->', agent_color), action.name,
            #           action.param_strings(), colored('at cost:', agent_color), action_cost,
            #           colored('at state:', agent_color), current_state.id, colored('resulting with state:', agent_color),
            #           next_state.id)

            if self.is_goal(next_state):
                return None, action.cost, True, action

            if State.has_public_changes(current_state, next_state):
                return next_state, action.cost, True, action

            return next_state, action.cost, False, action
        else:
            print('no possible actions for agent', self.name)
            return current_state, 0, False, None

    def print_Q_Table(self):
        for state in self.Q_values.keys():
            print(state.print_full_state())
            self.print_Q_table_state(state)

    def print_Q_table_state(self, state):
        agent_color = self.planner.agent_colors[self.id]

        # state_str = '\nstate: {0}'.format(str(state))
        # print(colored(state.print_full_state(), 'magenta'))
        for action in self.Q_values[state].keys():
            action_str = action.name + str(action.param_strings()) + \
                         ' | Q Value = {0}'.format(self.Q_values[state][action])
            print(colored(action_str, agent_color))

    # for debug
    def get_trajectory(self, traj_key):
        if traj_key in self._forward_trajectories:
            return self._forward_trajectories[traj_key]
        if traj_key in self._backward_trajectories:
            return self._backward_trajectories[traj_key]
        return None

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

    def resume_previous(self):
        """
        this function is used in drtdp to resume a previous trajectory that was waiting
        for QVALUE answers. This function should only be called by advance_trajectory.
        :return: True if we should break from the main loop in advance_trajectory else False
        """

        # restore saved data
        current_state = self.current_state
        next_state = self.next_state
        action = self.action
        traj = self.trajectory

        # finnish updating using saved data and other agents answers
        self.finnish_update(current_state, next_state, action, traj.trajectory_key)

        if self.is_goal(next_state):

            # next_state is the goal state - add traj to it's agent_trajectories here
            # since this trajectory will be removed and the state will not get updated with update function
            next_state.add_agent_trajectory(self.id, traj.trajectory_key)
            trajectory = self._forward_trajectories.pop(traj.trajectory_key)
            self._backward_trajectories[traj.trajectory_key] = trajectory

            self.planner.goal_reached(self.id, traj)

            if len(self.planner.agents) == 1:
                self.create_new_trajectory(self.initial_state)

            # break from main loop
            return True

        # if we have more than 1 agent, no matter what the next state is,
        # request Q values to choose the best agent to act
        # we do this instead of requesting on public changes to solve the issue of agents that can
        # perform private actions and never reach a public state change
        if len(self.planner.agents) > 1:
            # request best Q Value of next state from other agents
            self.planner.add_message(Message(QVALUE_REQ_MSG, next_state, self.id, traj.trajectory_key,
                                             prev_state=current_state, prev_action=action))

            # break from main loop - took out of public change for the case where an agent never gets to
            # perform a public action but can always perform private actions
            return True

        return False
