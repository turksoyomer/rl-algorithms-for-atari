class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def store(self, state, action, action_logprob, state_val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(action_logprob)
        self.state_values.append(state_val)
        self.rewards.append(reward)
        self.is_terminals.append(done)
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
