import gym
import time
from deeprl_hw2.dqn import DQNAgent

from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import model_from_json
from keras.optimizers import Adam

from deeprl_hw2.preprocessors import PreprocessorSequence, HistoryPreprocessor, AtariPreprocessor, NumpyPreprocessor
from deeprl_hw2.policy import GreedyEpsilonPolicy, UniformRandomPolicy


import sys
import json
from scipy import misc
import numpy as np

from gym.wrappers import Monitor


import matplotlib.pyplot as plt

def main():
    if(len(sys.argv) != 5):
        print("usage:{} <env> <model_json> <weights> <directory>".format(sys.argv[0]))
        return sys.exit()
    env = gym.make(sys.argv[1])
    env.frameskip = 1
    with open(sys.argv[2]) as json_file:
        model = model_from_json(json.load(json_file))
    model.load_weights(sys.argv[3])
    epsilon = 0.01
    input_shape = (84,84)
    history_size = 4
    eval_size = 1
    directory = sys.argv[4]

    history_prep = HistoryPreprocessor(history_size)
    atari_prep = AtariPreprocessor(input_shape,0,999)
    numpy_prep = NumpyPreprocessor()
    preprocessors = PreprocessorSequence([atari_prep, history_prep, numpy_prep]) #from left to right


    policy = GreedyEpsilonPolicy(epsilon)

    agent = DQNAgent(model, preprocessors, None, policy, 0.99, None,None,None,None)
    env = gym.wrappers.Monitor(env,directory,force=True)
    reward_arr, length_arr = agent.evaluate_detailed(env,eval_size,render=False, verbose=True)


    #check for preprocessors 
    # state = env.reset()

    # for i in range(0,5):
    #     process_state = preprocessors.process_state_for_network(state)
    #     state,reward,is_teminal,debug = env.step(0)
    # print(process_state.shape)
    # misc.imshow(process_state[:,:,0])
    # misc.imshow(process_state[:,:,1])
    # misc.imshow(process_state[:,:,2])

if __name__ == '__main__':
    main()