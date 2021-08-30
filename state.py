class State(object):

    # a dictionary mapping a literal (by name) to a list of literal assignments
    # for example: _literal_assignments['on'] -> [on,public,(e - public, a - public), on,public,(a - public, i - public),...]
    _literal_assignments = None
    _literals = None

    # a dictionary mapping agent id to the agents trajectory through this state and the cost of the trajectory
    # up to this state. for example - _agent_trajectories[1] --> (parent_state, cost)
    _agent_trajectories = None

    def __init__(self, literals, agent_private_keys, planner, initial=False):
        '''
        a class representing a state by the public facts (literals) and a mapping of agents to the keys of their
        private facts (literals)
        :param literals: a list of the public literals
        :param agent_private_keys: a dict mapping agent id's to a list of keys to their private facts (literals)
        '''

        # initialize state cost to 0
        self.cost = 0.0

        # for debug
        self.id = -1
        self.planner = planner

        self.is_initial = initial

        self._literals = sorted(literals, key=State._key_from_literal_name_and_params)
        self.agent_private_keys = agent_private_keys
        self.agent_possible_actions = {}
        self._literal_assignments = {}
        self._agent_trajectories = {}
        self.state_hash = None
        self.private = False

        for literal in literals:
            self._literal_assignments[literal.predicate.name] = self._literal_assignments.get(literal.predicate.name, [])
            self._literal_assignments[literal.predicate.name].append(literal)

            if literal.predicate.is_private:
                self.private = True

    @property
    def literals(self):
        return self._literals

    @property
    def agent_trajectories(self):
        return self._agent_trajectories

    def public_projection(self):
        return State(self.literals, {}, self.planner)

    def contains_public_state(self, other):
        '''
        compares the public state of self to the public state of other
        :param other: state object
        :return: True iff self's literals contain all of other's literals
        '''

        for literal in other.literals:
            if literal not in self.literals:
                return False

        return True

    def get_passing_agents_ids(self, exclude=None):
        '''
        :return: a list of pairs (id, list<trajectory keys>)
        '''

        passing_ids = []
        for agent_id in self.agent_trajectories.keys():
            if exclude is not None and not agent_id == exclude:
                passing_ids.append((agent_id, list(self.agent_trajectories[agent_id])))

        return passing_ids

    def get_common_literals(self, other_state):
        """
        :param other_state: state object
        :return: the number of public literals self has that are also in other_state
        """
        literals_count = 0
        for literal in self.literals:
            if literal in other_state.literals:
                literals_count += 1

        return literals_count

    def add_agent_trajectory(self, agent_id, trajectory_key):
        '''
        adds the trajectory key to the trajectories that the agent has that pass through this state
        :param agent_id: the id of the agent who's trajectory it is
        :param trajectory_key: the trajectory key
        '''

        agent_trajectories = self.agent_trajectories.get(agent_id, set())
        agent_trajectories.add(trajectory_key)
        self.agent_trajectories[agent_id] = agent_trajectories

    def possible_actions(self, agent_id, agent_type, agent_name, all_actions=None, agent_private_literals=None):
        '''
        returns a list of grounded actions that are possible (the preconditions are met)
        :param agent_private_literals: the first time this function is called it should get a dict of all private
                                       literals the agent has
        :param all_actions: the first time this function is called it should get a list of all actions
                            the agent has
        :param agent_id: the id of the agent that performs the actions
        :return: list<action> of grounded actions
        '''

        if agent_id not in self.agent_possible_actions:
            possible_actions = list(filter(lambda action: self.are_preconds_met(action, agent_id,
                                                                                agent_private_literals), all_actions))

            # if the first param type is the type of the agent - allow only the agent object
            # filter all actions that have first param that is not self
            possible_actions = list(filter(lambda action: self.is_first_param_agent(action, agent_type, agent_name),
                                           possible_actions))

            # create (action, expected total cost) pairs where the expected cost is initialized to 0.0
            # possible_actions = list(map(lambda action: (action, 0.0), possible_actions))
            self.agent_possible_actions[agent_id] = possible_actions

        # actions = list(map(lambda tup: tup[0], self.agent_possible_actions[agent_id]))
        return self.agent_possible_actions[agent_id]

    def is_first_param_agent(self, action, agent_type, agent_name):
        if len(action.params) == 0:
            return True
        first_param_type = action.params[0].type
        if first_param_type == agent_type:
            return action.params[0].value.value == agent_name
        return True

    def are_preconds_met(self, action, agent_id, agent_private_literals):
        '''
        returns True only if all the preconditions of action are met by the state
        :param action: an action object
        :param agent_id: the id of the agent to perform the action
        :return: True iff all the preconditions are met
        '''

        private_keys = self.agent_private_keys[agent_id]
        private_literals = [agent_private_literals[x] for x in private_keys]
        state_literals = self.literals + private_literals
        state_predicates = list(map(lambda lit: lit.predicate, state_literals))

        for precond in action.precond:

            # if a precondition is positive it has to exist in the state literals
            if (precond.is_positive()) and (precond not in state_literals):
                return False

            # if the precondition is negative it has to not exist in the state literals
            elif (not precond.is_positive()) and (precond.predicate in state_predicates):
                return False

        # all preconditions exist in the state, thus they are met
        return True

    @classmethod
    def has_public_changes(cls, current_state, next_state):

        curr_public = sorted(current_state.literals, key=State._key_from_literal_name_and_params)
        next_public = sorted(next_state.literals, key=State._key_from_literal_name_and_params)

        return not curr_public == next_public

    @classmethod
    def get_hash(cls, literals, agent_private_keys):

        public_literals_str = ''
        # print('before', literals)
        literals = sorted(literals, key=State._key_from_literal_name_and_params)
        # print('after', literals)

        for literal in literals:
            public_literals_str += str(literal)

        # print('before', agent_private_keys)
        agent_keys = [item for sublist in agent_private_keys.values() for item in sublist]
        agent_keys = sorted(list(map(lambda key: str(key), agent_keys)))
        # print('after', agent_keys)
        return hash(public_literals_str + str(agent_keys))

    def print_state(self):
        state_str = '\nstate ' + str(self.id) + ':\n'
        for key in self._literal_assignments.keys():
            state_str += str(key) + '\t--> '
            positive_literals = list(filter(lambda literal: literal.is_positive(), self._literal_assignments[key]))
            positive_literals = sorted(map(lambda literal: str(literal.predicate.arg_strings()), positive_literals))
            state_str += str(positive_literals) + '\n'
        # print(state_str)
        return state_str

    def print_full_state(self):
        state_str = self.print_state() + 'private keys:\n'
        for agent_id in self.agent_private_keys.keys():
            private_literals = []
            for literal_hash in self.agent_private_keys[agent_id]:
                private_literals.append(self.planner.agents[agent_id].private_literals[literal_hash])

            state_str += 'agent ' + str(agent_id) + '\t--> ' + str(private_literals) + '\n'

        # print(state_str)
        return state_str

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        state_str = ''
        for literal in self._literals:
            state_str += str(literal)

        return state_str

    def __hash__(self):
        if self.state_hash is None:
            state_hash = State.get_hash(self.literals, self.agent_private_keys)
            self.state_hash = state_hash

        return self.state_hash

    @staticmethod
    def _key_from_literal_name_and_params(literal):
        key_str = literal.predicate.name
        sorted_params = sorted(literal.predicate.args, key=lambda term: term.name)
        key_str = key_str + ''.join(map(str, sorted_params))
        return key_str


if __name__ == '__main__':
    from literal import Literal
    from predicate import Predicate
    from term import Term
    from copy import deepcopy

    literals1 = []
    for i in range(10):
        lit_name = 'literal{0}'.format(i)
        lit = Literal(Predicate(lit_name, private=False, args=[Term(name='a'), Term(name='b')]), True)
        literals1.append(lit)

    agent_keys1 = {0: [1, 2, 3],
                   1: [4, 5, 6]}
    agent_keys2 = {0: [1, 2, 3],
                   1: [4, 5, 7]}
    agent_keys3 = {1: [1, 2, 3],
                   0: [4, 5, 6]}

    hash1 = State.get_hash(literals1, agent_keys1)
    hash2 = State.get_hash(literals1, agent_keys2)

    # same public state, different private keys - should be False
    print('hash1 == hash2 is', hash1 == hash2)

    literals2 = deepcopy(literals1)
    lit_name = 'literal0'
    lit = Literal(Predicate(lit_name, private=False, args=[Term(name='b'), Term(name='a')]), True)
    literals2.append(lit)
    hash3 = State.get_hash(literals2, agent_keys1)
    hash4 = State.get_hash(literals2, agent_keys2)

    # different public state, same private keys - should be False
    print('hash2 == hash3 is', hash2 == hash3)

    # different public state, different private keys - should be False
    print('hash2 == hash4 is', hash2 == hash4)

    hash5 = State.get_hash(literals1, agent_keys3)

    # same public state, same agent keys but for other agents - should be True
    print('hash1 == hash5 is', hash1 == hash5)

