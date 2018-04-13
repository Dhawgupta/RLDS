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
from functions import one_hot_encoding, convertState,checkState
import pickle
import json
# import envOld
import time
import matplotlib.pyplot as plt
import argparse
import dictionaries
from NLU_model import NLU
from collections import defaultdict

# construct the argument parse and parse the arguments


##### The Argument Parser #########

# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
#                 help="the agent model to be loaded")
# ap.add_argument("-n","--nlu", required = True, help = "the nlu model to be loaded")


class NLU_simulator():
    def __init__(self, w1 = 1,w2 = 8,w3 = 13,nlu_name = "weight.h5"):
        self.state_size = 5
        self.actions = 14 # TODO 14 action are there but the action 0 i.e. greet will be superficial in training
        self.current_state = []  # the current state of the agent
        self.states = np.zeros([0, 5])  # this is the collection of all the states
        self.num_iter = 0
        self.w1 = w1  # interaction weight
        self.w2 = w2
        self.w3 = w3
        # 1/3/18 Keep a track of the actions taken i.e a count of number of action taken for each type
        self.actions_taken = np.zeros([self.actions])
        self.nlu_name = nlu_name
        # laods the nlu model
        self.nlu_model = NLU(filename =self.nlu_name ,load = True)
        self.real_slots_values = dict()
        self.guessed_slot_values = dict()
        # for k in dictionaries.tags2values:
        #     # we will randomyl assign a tag a value
        #     self.real_slots_values[k] = random.choice(dictionaries.tags2values[k])
        self.positions = {  # position of various slots
            "$DCITY$": 0,
            "$ACITY$": 1,
            "$DATE$": 2,
            "$TIME$": 3,
            "$CLASS$": 4

        }
        self.position2tags = {
            0:"$DCITY$",
            1:"$ACITY$",
            2:"$DATE$",
            3:"$TIME$",
            4:"$CLASS$"
        }

        self.init_state()

    def __repr__(self):
        """ representation of the environment  class"""
        # print self.guessed_slot_values
        # print self.real_slots_values
        # print self.current_state
        string = str(self.guessed_slot_values) + "\n" + str(self.real_slots_values) + "\n" + str(self.current_state)
        return string

    def init_state(self):
        self.states = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])  # initliase with zero for all values
        self.current_state = self.states[-1]
        # 1/3/18 reinit the actions taken matrix
        self.actions_taken = np.zeros([self.actions])
        self.threshold = 0.7
        self.guessed_slot_values = { "$ACITY$" : "Unknown",
                                     "$DCITY$" : "Unknown",
                                     "$DATE$"  : "Unknown",
                                     "$TIME$"  : "Unknown",
                                     "$CLASS$" : "Unknown"}

        for k in dictionaries.tags2values:
            # we will randomyl assign a tag a value
            self.real_slots_values[k] = random.choice(dictionaries.tags2values[k])

    def random_init_state(self):
        self.states = np.array(
            [[random.random() for i in range(self.state_size)]])  # initliase with zero for all values
        self.current_state = self.states[-1]
        # 1/3/18 reinit the actions taken matrix
        self.actions_taken = np.zeros([self.actions])
        self.guessed_slot_values = {"$ACITY$": "Unknown",
                                    "$DCITY$": "Unknown",
                                    "$DATE$": "Unknown",
                                    "$TIME$": "Unknown",
                                    "$CLASS$": "Unknown"}

        for k in dictionaries.tags2values:
            # we will randomyl assign a tag a value
            self.real_slots_values[k] = random.choice(dictionaries.tags2values[k])

    def reset(self):
        """
        reset the state to the statrting poitn
        """
        i = random.random()
        self.guessed_slot_values = {"$ACITY$": "Unknown",
                                    "$DCITY$": "Unknown",
                                    "$DATE$": "Unknown",
                                    "$TIME$": "Unknown",
                                    "$CLASS$": "Unknown"}
        if i <= 1: # no randomness
            self.init_state()
        else:
            self.random_init_state()
        self.num_iter = 0
        return self.current_state

    def check_slots(self):
        "Will cross check the guessesed aslots and real slots"
    def step(self,action):
        done = False

        reward = 0
        # action is a numeric value
        # TODO replace tags in reply with the real values

        # fixmed this part will come at the end because reply can change


        # print the user reply
        new_state = self.current_state.copy()
        if action == 13: # terminating action # todo dont update the state and done = true and give the reward comapre the guessed and real slots and decide a reward function
            done = True

            # fixme decide the reward over here
            # fixme still the reward is left to decide
            # dont change the new_state
            # reward = ???
            reply = dictionaries.indx2user_replies[action]
            # check if the implementation is correct or not
            all_correct = 1
            for i in range(5):
                if self.guessed_slot_values[self.position2tags[i]] != self.real_slots_values[self.position2tags[i]]:
                    all_correct = 0
            if all_correct == 1:
                reward = 1
            else:
                reward = -1


        elif action > 7 and action <= 12: # reask actions
            slot = action - 8 # give the slot number
            if self.guessed_slot_values[self.position2tags[slot]] == self.real_slots_values[self.position2tags[slot]]:
                reply = "yes"
                new_state[slot] = 1
            else:
                reply = "no"
                new_state[slot] = 0

        else: # action 0 to 7
            # [pred,prob] = self.nlu_model.parse_sentence(reply)
            reply = random.choice(dictionaries.indx2user_replies[action])
            for k in self.real_slots_values:
                if k in reply:
                    reply = reply.replace(k, self.real_slots_values[k])

            tags = defaultdict(list)
            prob_values = defaultdict(list)
            ourtags = []
            [pred, prob] = self.nlu_model.parse_sentence(reply)
            for t in pred:
                ourtags.append(dictionaries.labels2labels[t])

            print ourtags
            for i, val in enumerate(ourtags):
                if val in self.guessed_slot_values:
                    # guessed_slots_values[val] = reply.split(" ")[i]
                    # new_state[positions[val]] = prob[i]
                    tags[val].append(reply.split(" ")[i])
                    prob_values[val].append(prob[i])
            for k in prob_values:
                prob_values[k] = [sum(prob_values[k]) / len(prob_values[k])]

            # print tags
            # print prob_values
            for k in tags:
                self.guessed_slot_values[k] = ' '.join(tags[k])
                new_state[self.positions[k]] = prob_values[k][0]


            # finished else
        agent_sentence = dictionaries.actions_sentences[action]
        # reply = random.choice(dictionaries.indx2user_replies[action])
        for k in self.guessed_slot_values:
            if k in agent_sentence:
                agent_sentence = agent_sentence.replace(k, self.guessed_slot_values[k])

        print("\n\nAgent >{}".format(agent_sentence))
        print("User  >{}\n\n".format(reply))

        # calculate the apropiate reward
        # reward = 0
        self.current_state = np.array([float('%.3f' % elem) for elem in new_state])
        self.states = np.append(self.states, [self.current_state], axis=0)
        return [self.current_state,reward,done]




































# # greeting the agent
# print("Agent > {}".format(dictionaries.actions_sentences[0])) # greet the user
# reply = raw_input("User >").rstrip().lower()
# # reply = reply.lower()

# now we will feed the reply in the NLU model
# we feed the input and get the tags approximat
# the predicted, prob values
# state = [0 for i in range(state_size)]
# state = tuple(state)
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
# print state
# action = agent.chooseAction(convertState(state))
# while not (action == 13) :
#     # continue the loop
#     # prepare a dicitonary of the tages and append them in the list
#     tags = defaultdict(list)
#     prob_values = defaultdict(list) # store the prbaiiltty value sof corrspoding tages
#     question = dictionaries.actions_sentences[action]
#     for k in guessed_slots_values:
#         if k in question:
#             question = question.replace(k,guessed_slots_values[k])
#
#     print("Agent >{}".format(question))  # greet the user
#     # reply = raw_input("User >").rstrip().lower()
#     reply = dictionaries.indx2user_replies[action][0]
#
#     for t in dictionaries.tags2values:
#         if t in reply:
#             reply = reply.replace(t,dictionaries.tags2values[t][0])
#     new_state = [state[i] for i in range(state_size)]
#     ourtags = []
#     if action > 7 and action <= 12:
#         if "yes" in reply:
#             new_state[action - 7] = 1
#         else:
#             new_state[action -7] = 0
#
#     elif action == 1:
#         # later we can integrate word embedddings
#         if len(reply.split(" ")) == 1:
#             reply = "from " + reply
#     elif action == 2:
#         # later we can integrate word embedddings
#         if len(reply.split(" ")) == 1:
#             reply = "to " + reply
#
#
#
#     [pred, prob] = nlu_model.parse_sentence(reply)
#     for t in pred:
#         ourtags.append(dictionaries.labels2labels[t])
#
#     for i, val in enumerate(ourtags):
#         if val in guessed_slots_values:
#             # guessed_slots_values[val] = reply.split(" ")[i]
#             # new_state[positions[val]] = prob[i]
#             tags[val].append(reply.split(" ")[i])
#             prob_values[val].append(prob[i])
#     for k in prob_values:
#         prob_values[k] = [sum(prob_values[k])/len(prob_values[k])]
#
#     # print tags
#     # print prob_values
#     for k in tags:
#         guessed_slots_values[k] = ' '.join(tags[k])
#         new_state[positions[k]] = prob_values[k][0]
#
#     if action == 0:
#         new_state[0] = 1
#     new_state[-1] += 1
#     # print guessed_slots_values
#     # print ourtags
#     print new_state  # start state
#     state = tuple(new_state)
#     action = agent.chooseAction(convertState(state))
#     # print action


# def mean(lis):
    # input is a list and return th mean
