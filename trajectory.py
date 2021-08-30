class Trajectory(object):

    def __init__(self, first_state, agent_id, trajectory_key, planner, parent_agent_id=None, parent_trajectory_key=None):
        '''
        a class representing a trajectory of an agent.
        the trajectory can begin as a continuation of another agent's trajectory.
        the trajectory uses a list of (state, action) pairs called path, the action is what resulted in state,
        if the action is None -> either state is the initial state or the action was performed by another agent.
        :param first_state: the state the trajectory begins with
        :param agent_id: the id of the agent who's trajectory it is
        :param trajectory_key: the key identifying the trajectory for the agent (!) should be encrypted
        :param parent_agent_id: the id of the agent that caused the creation of the trajectory or None if it wasn't
        caused by another agent
        :param parent_trajectory_key: the trajectory key identifying the trajectory for the parent agent
        '''

        self.path = [(first_state, None)]

        # identification of the agent who's trajectory it is
        self.agent_id = agent_id
        self.trajectory_key = trajectory_key

        # identification of the parent agent that might take part in the trajectory
        self.parent_agent_id = parent_agent_id
        self.parent_trajectory_key = parent_trajectory_key

        # for debug
        self.planner = planner

    # todo - maybe rolling hash on the states and actions
    def hash_path(self):
        print('trajectory - hash_path not yet implemented')
        exit(-1)

    def __eq__(self, other):
        return self.agent_id == other.agent_id and self.trajectory_key == other.trajectory_key and \
               self.parent_agent_id == other.parent_agent_id and \
               self.parent_trajectory_key == other.parent_trajectory_key and \
               self.hash_path() == other.hash_path()

    def current_state(self):
        '''
        :return: the last state reached in this trajectory
        '''
        state_index = len(self.path) - 1
        state = self.path[state_index][0]
        return state

    def set_current_action(self, action):
        '''
        set the action performed at the last state reached in this trajectory
        :param action: an action object
        '''

        current_state_index = len(self.path) - 1
        current_state = self.path[current_state_index][0]

        # remove the (state, action) pair
        self.path.pop(current_state_index)

        # set the new action state pair
        self.path.append((current_state, action))

    def add_state_action(self, state, action):
        '''
        adds the new state and action to the trajectory path,
        if the state is the same, just update the action
        :param state: the new state reached by the trajectory
        :param action: the action performed at the state
        '''

        if self.current_state().id != state.id:
            self.path.append((state, action))
        else:
            cur_state, cur_action = self.path.pop()
            self.path.append((cur_state, action))

    def print_path(self):

        if self.parent_trajectory_key is not None:
            parent_agent = self.planner.agents[self.parent_agent_id]
            parent_traj = parent_agent.get_trajectory(self.parent_trajectory_key)
            parent_traj.print_path()

        print('----------- path of traj', str(self.trajectory_key), '-----------')

        path_str = ''
        for tup in self.path:
            state = tup[0]
            path_str += 'in state' + str(state.id) + ' | '
            action = tup[1]
            if action is not None:
                params = list(map(lambda term: str(term), action.params))
                path_str += action.name + ' ' + str(params) + '\n'

        last_state = self.path[len(self.path) - 1][0]
        print(path_str)
        last_state.print_state()

    def get_common_literals_count(self, state):
        '''
        counts the literals state and the current state in the path have in common
        :param state: a state to compare to
        :return: the number of literals in common
        '''

        count = 0
        for literal in state.literals:
            if literal in self.current_state().literals:
                count += 1
        return count
