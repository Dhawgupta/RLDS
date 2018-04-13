
import numpy as np
import sys
import random
import time
from functions import one_hot_encoding, convertState
import pickle
import json
# import envOld
import time
from DQN1 import DQNAgent
import matplotlib.pyplot as plt
import argparse
import env1
import dictionaries
from NLU_model import NLU
from collections import defaultdict
from NLU_env import NLU_simulator
from functions import one_hot_encoding

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="the agent model to be loaded")
ap.add_argument("-n","--nlu", required = True, help = "the nlu model to be loaded")
args = vars(ap.parse_args())

env = NLU_simulator(nlu_name = args["nlu"])

model_file = args["model"]
print(args["model"],args["nlu"])

# print("Loading Model from : {}".format(model_file))
# agent = DQNAgent
# data = dict()
# with open(model_file,'r') as handle:
#     data = pickle.load(handle)
# # fixme the trained table model is for 13 actions excluding the greetin whereas the environemnt is based on 14 actions which include greeting, keep this in mind
action_size = 13
state_size = 5
# agent = Qlearn([i for i in range(action_size)], epsilon= 0 , alpha=0.2, gamma = 0.9)
agent = DQNAgent(state_size, action_size, hiddenLayers=[75], dropout=0.0, activation='relu',loadname = args["model"], saveIn = False, learningRate = 0.05, discountFactor=0.7,epsilon = 0.001)
print("Loading FIle {}".format(args["model"]))
agent.load(model_file)

# agent.q = data.copy()

# state = env.reset()
# fixme currently working with 0.1 resolution
done = False
action = 0
state = env.reset()
all_act = []
for z in range(0,action_size):
    all_act.append(z)
while not done:
    print ("State: {}".format(state))
    [next_state,reward,done] = env.step(action)
    state = next_state
    state = np.reshape(state,[1,len(state)])
    action = agent.act(state, all_act)
    print("#####The action {}".format(action) )
    # action = agent.chooseAction(convertState10(state))
    action = action +1
