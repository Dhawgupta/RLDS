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
"""

import numpy as np
import random
from functions import *

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
    def __init__(self,w1 = 1, w2 = 8, w3 = 13):
        """
        The weights are as follows
        w1 : is the interaction weigt
        w2 : is the change in confidence value
        w3: is the weight given to weight value
        """
        self.state_size = 5
        self.actions = 13
        self.current_state = [] # the current state of the agent
        self.states = np.zeros([0,5]) # this is the collection of all the states
        self.init_state()
        self.num_iter = 0
        # self.actions = ['greet','askDeptCity','askArrCity','askDepDay','askDepTime','askClass','askDeptArrCity','askDateTIme','reaskDeptCity','reaskArrCity','reaskDepDay','reaskDepTime','reaskClass','closeConversation' ]
        self.w1 = w1 # interaction weight
        self.w2 = w2
        self.w3 = w3
        # 1/3/18 Keep a track of the actions taken i.e a count of number of action taken for each type
        self.actions_taken = np.zeros([self.actions]) 



    def init_state(self):
        self.states = np.array([[0.0 , 0.0 ,0.0 ,0.0, 0.0]]) # initliase with zero for all values
        self.current_state = self.states[-1]
        # 1/3/18 reinit the actions taken matrix
        self.actions_taken = np.zeros([self.actions])
        self.threshold = 0.7

    def random_init_state(self):
        self.states = np.array([[random.random() for i in range(self.state_size)]])  # initliase with zero for all values
        self.current_state = self.states[-1]
        # 1/3/18 reinit the actions taken matrix
        self.actions_taken = np.zeros([self.actions])
    def step(self,action):
        # this will be the next step to take for an action lies between 0 - 12 including both
        """
        Now the confidence value of the slots will be defined on the language unit of the system:
        We will assume the langugae system to be a random funciton generator which
        """
        done = False

        reward = 0
        # action is a numeric value

        new_state = np.array([float('%.4f' % elem) for elem in self.current_state])
        # if action == 0 and self.num_iter == 0 : # greeting should be first
        #     new_state[0] = 1
        if action >= 0 and action <= 4:# the action is to ask a slot value
            new_state[action] = 0.2*random.random() + 0.6 # the confidenec value lies between 0.6 - 0.8
        if action > 6  and action <=11:
            # this will seq increase the confidence value action 8 - 7
            # @1/3/18 18:49:00 Update : reask will only assign confidence once the slot has been asked for
            if new_state[action -7] < 0.2:  # @1/3/18 18:49:00this implies that the ask part has not happened
                pass 
            else:
                new_state[action - 7] = (1 - new_state[action - 7])*0.85 + new_state[action - 7] # @1/3/18 18:49:00the ask part has happned
        if action == 5 :
            new_state[1] = 0.2 *random.random() + 0.6
            new_state[2] = 0.2 *random.random() + 0.6
        if action == 6:
            new_state[3] = 0.2 *random.random() + 0.6
            new_state[4] = 0.2 *random.random() + 0.6
        # else:
        #     print("Final Step")

        # @1/3/18 Increase the action taken value for the action taken 


        # self.actions_taken[action] += 1


        # if action == 0 and self.num_iter == 0:  # reward for greeting
        #     # @1/3/18 adding a negative reward for not greeting
        #     reward = self.w3
        # elif self.num_iter == 0 and action != 0:
        #     reward = -self.w3 # @1/3/18 giving the negative reward for not greeting
        if action == 12: # final action
            done = True # set the sim finished
            # 28/2/18 impleted the soft check functionalt
            val =checkState(self.current_state)

            if val :
                """
                The returned value will be negative if less than 3 states are satifies
                """
                reward =  (self.w2*(sum(self.current_state))) #  1/3/18 removing the greeting from final reward + self.w3*self.current_state[0])
            else:
                reward  =  -self.w2*(sum(np.ones(5)) - sum(self.current_state))  # 1/3/18 removing the reward from final greeting + self.w3*self.current_state[0]
        else:
            # @1/3/18 we have to update the self.w1 to be affected byt he action taken 
            reward = self.w2*(sum(new_state) - sum(self.current_state)) - self.w1 # @1/3/18 a factor of action_taken is given

        

        # 28/2/18 add a noise to the state
        # noise_tobe_added = np.random.rand(len(self.current_state))*0.02 -0.01
        # 28/2/18 now we will remove noise from greeting and iterations
        # noise_tobe_added[0] = 0 # greet state to be zero
        # noise_tobe_added[-1] = 0 # iteratiion state to be zero
        self.num_iter = self.num_iter + 1
        # new_state[6] = self.num_iter
        self.current_state = np.array([float('%.3f' % elem) for elem in new_state])
        
        # adding the noise to the current state
        # self.current_state = self.current_state + noise_tobe_added

        self.states = np.append(self.states, [self.current_state], axis=0)

       
        return self.current_state,reward, done # return the state and reward
        # at the moment we will consider the end episode to return the reward
    	# need an indentifier to indetify the end of DialougeSimualtion
    def reset(self):
        """
        reset the state to the statrting poitn
        """
        i = random.random()
        # TODO for random state out 1 if not random state required
        if i < 0.5:
            self.init_state()
        else:
            self.random_init_state()
        self.num_iter = 0
        return self.current_state
