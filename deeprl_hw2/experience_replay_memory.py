
from deeprl_hw2.core import ReplayMemory, Sample
import numpy as np
import sys
import copy

class ActionSample(Sample):
    """
    Implementation of the Sample Class 
    """
    def __init__(self, state, next_state, action, reward):
        """
        Initialization, we assumed the state is already preprocessed
        """
        self._state = state
        self._next_state = next_state
        self._action = action
        self._reward = reward


class ExperienceReplayMemory(ReplayMemory):

    def __init__(self, max_size):
        """
        Set up the replay memory
        """
        self._max_size = max_size
        self._full = False
        self._initialized = False
        self.clear()

    def __len__(self):
        return self._max_size if self._full else self._index

    def size(self):
        return sys.getsizeof(self._memory)

    def append(self, cur_state, next_state, action, reward, is_terminal):

        #make sure they are valid first
        action = None if is_terminal else np.uint8(action)
        next_state = None if is_terminal else np.uint8(next_state)
        reward = None if is_terminal else np.int16(reward)

        new_sample = ActionSample(np.uint8(cur_state), next_state, action, reward)
        self._memory[self._index] = new_sample
        self._index += 1
        #if we reach the end of the index, loop back
        if self._index >= self._max_size:
            self._full = True
            self._index = self._index % self._max_size


    def sample(self, batch_size, indicies=None):
        #generate the indicies
        if(indicies == None):
            if self._full:
                indicies = (np.random.randint(0,self._max_size, size=batch_size)).tolist()
            elif self._index != 0:
                indicies = (np.random.randint(0,self._index, size=batch_size)).tolist()
            else:
                indicies = []

        #the return sample will be in the form of [state, nxt_state, reward, action]
        curr_state_list = []
        next_state_list = []
        action_list = []
        reward_list = []
        for i in indicies:
            #get sample
            sample = self._memory[i]

            #ignore if it's terminal
            while sample._next_state is None:
                i = (i - 1)%self._max_size
                sample = self._memory[i]


            #add the sample to the list
            curr_state_list.append(sample._state)
            next_state_list.append(sample._next_state)
            action_list.append(sample._action)
            reward_list.append(sample._reward)

        return np.array(curr_state_list), np.array(next_state_list), np.array(reward_list), np.array(action_list)

    def clear(self):

        self._filled_size = 0
        self._index = 0 
        self._full = False
        self._state_size = None

        if(not self._initialized):
            #We create a random sample here to play with
            state = np.random.randint(0,100,(84,84))
            _random_sample = ActionSample(state.astype(np.uint8),state.astype(np.uint8),np.uint8(1), np.int16(1))
            #initialize the memory by copying everthing
            self._memory = [copy.deepcopy(_random_sample) for x in range(self._max_size)]
            self._initialized = True
