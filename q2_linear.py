
import gym
import time
from deeprl_hw2.dqn import DQNAgent

from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import Model, Sequential
from keras.optimizers import Adam

from deeprl_hw2.preprocessors import PreprocessorSequence, HistoryPreprocessor, AtariPreprocessor
from deeprl_hw2.policy import LinearDecayGreedyEpsilonPolicy, UniformRandomPolicy
from deeprl_hw2.action_replay_memory import ActionReplayMemory
from deeprl_hw2.objectives import huber_loss
from deeprl_hw2.utils import memory_burn_in

import numpy as np
import json
import sys

def create_model(window, input_shape, num_actions,
                 model_name='q_network'):

    model = Sequential(name=model_name)
    model.add(Flatten(input_shape=(input_shape[0],input_shape[1],window)))
    model.add(Dense(units=num_actions, activation='linear'))
    return model

def main():

    #env = gym.make("Enduro-v0")
    #env = gym.make("SpaceInvaders-v0")
    #env = gym.make("Breakout-v0")

    model_name = "q2"
    if(len(sys.argv) >= 2):
        model_name = sys.argv[1]

    if(len(sys.argv) >= 3):
        env = gym.make(sys.argv[2])
    else:
        #env = gym.make("Enduro-v0")
        env = gym.make("SpaceInvaders-v0")
        #env = gym.make("Breakout-v0")

    #no skip frames
    env.frameskip = 1

    input_shape = (84,84)
    batch_size = 1
    num_actions = env.action_space.n 
    memory_size = 1
    memory_burn_in_num = 1
    start_epsilon = 1
    end_epsilon = 0.01
    decay_steps = 1000000
    target_update_freq = 1 #no targeting
    train_freq = 4 #How often you train the network
    history_size = 4
    
    history_prep = HistoryPreprocessor(history_size)
    atari_prep = AtariPreprocessor(input_shape,0,999)
    preprocessors = PreprocessorSequence([atari_prep, history_prep]) #from left to right

    policy = LinearDecayGreedyEpsilonPolicy(start_epsilon, end_epsilon,decay_steps)

    linear_model = create_model(history_size, input_shape, num_actions,model_name)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    loss_func = huber_loss
    #linear_model.compile(optimizer, loss_func)

    random_policy = UniformRandomPolicy(num_actions)
    #memory = ActionReplayMemory(1000000,4)
    memory = ActionReplayMemory(memory_size, history_size)
    #memory_burn_in(env,memory,preprocessors,memory_burn_in_num,random_policy)

    #print(reward_arr)
    #print(curr_state_arr)
    agent = DQNAgent(linear_model, preprocessors, memory, policy,0.99, target_update_freq,None,train_freq,batch_size)
    agent.compile(optimizer, loss_func)
    agent.save_models()
    agent.fit(env,1000000,100000)
    # #agent.evaluate(env, 5)

if __name__ == '__main__':
    main()
