import gym
import time
from deeprl_hw2.dqn import DQNAgent

from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import model_from_json
from keras.optimizers import Adam

from deeprl_hw2.preprocessors import PreprocessorSequence, HistoryPreprocessor, AtariPreprocessor
from deeprl_hw2.policy import GreedyEpsilonPolicy, UniformRandomPolicy
from deeprl_hw2.action_replay_memory import ActionReplayMemory

import sys
import json
from scipy import misc
import numpy as np

def main():
    if(len(sys.argv) != 6):
        print("usage:{} <env> <model_json> <weights> <render> <random>".format(sys.argv[0]))
        return sys.exit()
    env = gym.make(sys.argv[1])
    env.frameskip = 1
    with open(sys.argv[2]) as json_file:
        model = model_from_json(json.load(json_file))
    model.load_weights(sys.argv[3])
    epsilon = 0.01
    input_shape = (84,84)
    history_size = 4
    eval_size = 100
    render = (sys.argv[4] == "y")

    history_prep = HistoryPreprocessor(history_size)
    atari_prep = AtariPreprocessor(input_shape, 0, 999)
    preprocessors = PreprocessorSequence([atari_prep, history_prep]) #from left to right

    if(sys.argv[5] == "y"):
        print("using random policy")
        policy = UniformRandomPolicy(env.action_space.n)
    else:
        print("using greedy policy")
        policy = GreedyEpsilonPolicy(epsilon)

    agent = DQNAgent(model, preprocessors, None, policy, 0.99, None,None,None,None)
    reward_arr, length_arr = agent.evaluate_detailed(env,eval_size,render=render, verbose=True)
    print("\rPlayed {} games, reward:M={}, SD={} length:M={}, SD={}".format(eval_size, np.mean(reward_arr),np.std(reward_arr),np.mean(length_arr), np.std(reward_arr)))
    print("max:{} min:{}".format(np.max(reward_arr), np.min(reward_arr)))


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