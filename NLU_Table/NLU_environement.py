"""
created : 13.3.18
The interaction between the NLU model and the agent andetc
Table interaaction at the moment

The functions for environenent are in teh NLU_env_functions.py
"""



import numpy as np
import sys
import random
import time
from functions import one_hot_encoding, convertState
import pickle
import json
# import envOld
import time
from qlearn_table import Qlearn
import matplotlib.pyplot as plt
import argparse
import dictionaries
from NLU_model import NLU
from collections import defaultdict

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="the agent model to be loaded")
ap.add_argument("-n","--nlu", required = True, help = "the nlu model to be loaded")


args = vars(ap.parse_args())

# define the state and action sizes
state_size = 5 # for the table using the old
action_size  = 13

# getting the file and the Q table

filename = args["model"]
print("The loading File {}".format(filename))


#### loading the models
# getting the q table data from the file
data = dict()
with open(filename, 'r') as handle:
    data = pickle.load(handle)


# init the agent which will give us the action
agent = Qlearn([i for i in range(action_size)], epsilon= 0 , alpha=0.2, gamma = 0.9)
# assining the data to the agnet
agent.q = data.copy()




nlu_model = NLU(filename =args["nlu"] ,load = True)
###### models loaded ######

# this dict will contins the corresponding slots value guessed by the nlu model
guessed_slots_values = dict()
guessed_slots_values["$ACITY$"] = "Unknown"
guessed_slots_values["$DCITY$"] = "Unknown"
guessed_slots_values["$DATE$"] = "Unknown"
guessed_slots_values["$TIME$"] = "Unknown"
guessed_slots_values["$CLASS$"] = "Unknown"

positions = {
    "$DCITY$":1,
    "$ACITY$":2,
    "$DATE$": 3,
    "$TIME$": 4,
    "$CLASS$": 5,

}

# # greeting the agent
# print("Agent > {}".format(dictionaries.actions_sentences[0])) # greet the user
# reply = raw_input("User >").rstrip().lower()
# # reply = reply.lower()

# now we will feed the reply in the NLU model
# we feed the input and get the tags approximat
# the predicted, prob values
state = [0 for i in range(state_size)]
state = tuple(state)
# [pred, prob] = nlu_model.parse_sentence(reply)
# ourtags = []
# for t in pred:
#     ourtags.append(dictionaries.labels2labels[t])
#
# # this the tages that we want
# for i,val in enumerate(ourtags):
#     if val in guessed_slots_values:
#         guessed_slots_values[val] = reply.split(" ")[i]
#         state[positions[val]] = prob[i]
#
# print guessed_slots_values
# print ourtags
# print state # start state


# now we make the loop
print state
action = agent.chooseAction(convertState(state))
while not (action == 13) :
    # continue the loop
    # prepare a dicitonary of the tages and append them in the list
    tags = defaultdict(list)
    prob_values = defaultdict(list) # store the prbaiiltty value sof corrspoding tages
    question = dictionaries.actions_sentences[action]
    for k in guessed_slots_values:
        if k in question:
            question = question.replace(k,guessed_slots_values[k])

    print("Agent >{}".format(question))  # greet the user
    # reply = raw_input("User >").rstrip().lower()
    reply = dictionaries.indx2user_replies[action][0]

    for t in dictionaries.tags2values:
        if t in reply:
            reply = reply.replace(t,dictionaries.tags2values[t][0])
    new_state = [state[i] for i in range(state_size)]
    ourtags = []
    if action > 7 and action <= 12:
        if "yes" in reply:
            new_state[action - 7] = 1
        else:
            new_state[action -7] = 0

    elif action == 1:
        # later we can integrate word embedddings
        if len(reply.split(" ")) == 1:
            reply = "from " + reply
    elif action == 2:
        # later we can integrate word embedddings
        if len(reply.split(" ")) == 1:
            reply = "to " + reply



    [pred, prob] = nlu_model.parse_sentence(reply)
    for t in pred:
        ourtags.append(dictionaries.labels2labels[t])

    for i, val in enumerate(ourtags):
        if val in guessed_slots_values:
            # guessed_slots_values[val] = reply.split(" ")[i]
            # new_state[positions[val]] = prob[i]
            tags[val].append(reply.split(" ")[i])
            prob_values[val].append(prob[i])
    for k in prob_values:
        prob_values[k] = [sum(prob_values[k])/len(prob_values[k])]

    # print tags
    # print prob_values
    for k in tags:
        guessed_slots_values[k] = ' '.join(tags[k])
        new_state[positions[k]] = prob_values[k][0]

    if action == 0:
        new_state[0] = 1
    new_state[-1] += 1
    # print guessed_slots_values
    # print ourtags
    print new_state  # start state
    state = tuple(new_state)
    action = agent.chooseAction(convertState(state))
    # print action


# def mean(lis):
    # input is a list and return th mean
