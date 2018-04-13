# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 06:40:03 2017

@author: dawg
Last File : envOld2
Changes made:
1. Going to implement the action taken counter
2. Giving reward for greeting in the starting with negative reward
3. 
all changes with comment @1/3/18

Updated after @6/3/18

Previous file : env.py picking up from tulika maams work

Things to be implemented
1. the positive reward to 1
2. Cap for the negative reward
    2,1 w2*threshold*5
    2.2 w3
3. Added the significa  of w2 again
previous file is envOld2.py
"""

import numpy as np
import random
from functions1 import *


class DialougeSimulation(object):
    """
    State Descipriton:
    State: contains 6 variables (greet, deptCity, arrCity, deptTime, depDay, class ,number of iteration uptil no)
     Action Set :
     greet (0), ask{arrCity,deptCity,deptTime,depDay,class}()
     reask/confirm{arrCity,deptCity,deptTime,depDay,class} .
     askDeptandArr, askDateTime # hybrid actions

    # adding a seveth parameter for number of iterations
    """

    # @1/3/18 changing the default weight value of w2 to 8 and w3 to 13
    def __init__(self, w1=1, w2=8, w3=13):
        """
        The weights are as follows
        w1 : is the interaction weigt
        w2 : is the change in confidence value
        w3: is the weight given to weight value
        """
        self.state_size = 5
        self.actions = 13
        self.current_state = []  # the current state of the agent
        self.states = np.zeros([0, 5])  # this is the collection of all the states
        self.init_state()
        self.num_iter = 0
        # self.actions = ['greet','askDeptCity','askArrCity','askDepDay','askDepTime','askClass','askDeptArrCity','askDateTIme','reaskDeptCity','reaskArrCity','reaskDepDay','reaskDepTime','reaskClass','closeConversation' ]
        self.w1 = w1  # interaction weight
        self.w2 = w2
        self.w3 = w3
        # 1/3/18 Keep a track of the actions taken i.e a count of number of action taken for each type
        self.actions_taken = np.zeros([self.actions])
        # @6/3/18 calculate the max possible value
        self.threshold = 0.7
        #self.max_reward = max(self.w3, self.w2 * 5 * self.threshold)

    def init_state(self):
        self.states = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])  # initliase with zero for all values
        self.current_state = self.states[-1]
        # 1/3/18 reinit the actions taken matrix
        self.actions_taken = np.zeros([self.actions])

    def random_init_state(self):
        self.states = np.array(
            [[random.random() for i in range(self.state_size)]])  # initialise with zero for all values
        self.current_state = self.states[-1]
        # 1/3/18 reinit the actions taken matrix
        self.actions_taken = np.zeros([self.actions])


    def check(self, action):
        self.threshold = 0.7
        if (self.threshold < self.current_state[action - 7 - 1]):
            return 1
        else:
            return 0

    def step(self, action):
        # this will be the next step to take for an action lies between 0 - 12 including both
        """
        Now the confidence value of the slots will be defined on the language unit of the system:
        We will assume the langugae system to be a random funciton generator which
        """
        done = False

        reward = 0
        # action is a numeric value

        new_state = np.array([float('%.4f' % elem) for elem in self.current_state])
        #if action == 0 and self.num_iter == 0:  # greeting should be first
            #new_state[0] = 1
        if action >= 0 and action <= 4:  # the action is to ask a slot value
            new_state[action] = 0.2 * random.random() + 0.55  # the confidenec value lies between 0.6 - 0.8
        if action > 6 and action <= 11:
            # this will seq increase the confidence value action 8 - 7
            # @1/3/18 18:49:00 Update : reask will only assign confidence once the slot has been asked for
            if new_state[action - 7] < 0.1:  # @1/3/18 18:49:00this implies that the ask part has not happened
                pass
            else:
                new_state[action - 7] = (1 - new_state[action - 7]) * 0.85 + new_state[
                    action - 7]  # @1/3/18 18:49:00the ask part has happned
        if action == 5:
            new_state[0] = 0.2 * random.random() + 0.55
            new_state[1] = 0.2 * random.random() + 0.55
        if action == 6:
            new_state[2] = 0.2 * random.random() + 0.55
            new_state[3] = 0.2 * random.random() + 0.55

        # @1/3/18 Increase the action taken value for the action taken 
        # self.actions_taken[action] += 1

        #if action == 0 and self.num_iter == 0:  # reward for greeting
            # @1/3/18 adding a negative reward for not greeting
            #reward = self.w3
        #elif self.num_iter == 0 and action != 0:
            #print("w3=", self.w3)
            #reward = -self.w3  # @1/3/18 giving the negative reward for not greeting
        #elif self.num_iter != 0 and action == 0:
            #reward = -self.w3
        if action == 12:
            done = True  # set the sim finished
            # 28/2/18 impleted the soft check functionalt
            val = soft_check_state(self.current_state)

            if val:
                """
                The returned value will be negative if less than 3 states are satifies
                """
                # @7/3/18 added w2 again
                reward = val * (sum(self.current_state))*self.w2  # 1/3/18 removing the greeting from final reward + self.w3*self.current_state[0])
                reward = abs(reward)
                print("r in if=" + str(reward))
            else:
                # @7/3/18 added w2 again
                reward = -self.w2*(sum(np.full(5, 1)) - sum(self.current_state))  # 1/3/18 removing the reward from final greeting + self.w3*self.current_state[0]
                print("r in else=" + str(reward))
        else:
            # @1/3/18 we have to update the self.w1 to be affected byt he action taken 
            print(new_state)
            print(self.current_state)
            # @7/3/18 added w2 again
            reward = self.w2*(sum(new_state) - sum(self.current_state))  # - self.w1*self.actions_taken[action] # @1/3/18 a factor of action_taken is given
            #print(reward)
            #print(self.actions_taken[action])
            reward = reward - self.w1 
            # val = new_check_state(self.current_state)
            # print("val=" + str(val))
            # reward = reward + val


        self.actions_taken[action] += 1
        # 28/2/18 add a noise to the state
        noise_tobe_added = np.random.rand(len(self.current_state))*0.02 -0.01
        # 28/2/18 now we will remove noise from greeting and iterations
        # noise_tobe_added[0] = 0 # greet state to be zero
        # noise_tobe_added[-1] = 0 # iteratiion state to be zero
        self.num_iter = self.num_iter + 1
        #new_state[6] = self.num_iter
        self.current_state = np.array([float('%.3f' % elem) for elem in new_state])

        # adding the noise to the current state
        # self.current_state = self.current_state + noise_tobe_added

        self.states = np.append(self.states, [self.current_state], axis=0)
        reward = float(reward)#/self.max_reward

        return self.current_state, reward, done  # return the state and reward
        # at the moment we will consider the end episode to return the reward

    # need an indentifier to indetify the end of DialougeSimualtion
    def reset(self):
        """
        reset the state to the statrting poitn
        """
        i = random.random()
        if i < 1.2:
            self.init_state()
        else:
            self.random_init_state()
        self.num_iter = 0

        return self.current_state
