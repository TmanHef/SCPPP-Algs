PUBLIC_STATE_MSG = 'public state reached message'
BACK_UPDATE_MSG = 'goal state reached message'
QVALUE_REQ_MSG = 'Q Value request message'
QVALUE_ANS_MSG = 'Q Value answer message'


class Message(object):

    def __init__(self, type, state, gen_agent_id, trajectory_key,  rec_agent_id=None, q_value=None,
                 prev_state=None, prev_action=None):
        '''
        :param type: PUBLIC_STATE_MSG / BACK_UPDATE_MSG
        :param state: a state object
        :param gen_agent_id: the id of the agent who generated the message
        :param rec_agent_id: if type is BACK_UPDATE_MSG - the id of the agent who's trajectory led to the state
        :param trajectory_key: if type is PUBLIC_STATE_MSG - this should be the key of gen_agent_id's trajectory
                               if type is BACK_UPDATE_MSG - this should be the key of rec_agent_id's trajectory
        '''

        self.type = type
        self.state = state
        self.gen_agent_id = gen_agent_id
        self.rec_agent_id = rec_agent_id
        self.trajectory_key = trajectory_key
        self.Q_Value = q_value
        self.prev_state = prev_state
        self.prev_action = prev_action
