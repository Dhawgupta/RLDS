import random
from collections import deque
# FIXME this has the average implementatino

class Qlearn:
    def __init__(self,actions,epsilon = 0.1, alpha = 0.2, gamma = 0.9):
        """

        :param actions:
        :param epsilon:
        :param alpha:
        :param gamma:
        """
        self.q = {} # the dictionary for q function
        self.epsilon = epsilon # exploraiton
        self.alpha = alpha # discount rate
        self.gamma = gamma
        self.actions = actions
        self.memory1 = deque(maxlen=100)

    def rem_rew(self,reward):
        self.memory1.append((reward))

    def avg_rew(self):
        sum1 = 0
        i=0
        for elem in self.memory1:
            i = i + 1
            sum1 = sum1 + elem
        avr = sum1/i
        return avr


    def getQ(self, state,action): # to return the q value correspoding to an state action state
        return self.q.get((state,action), 0.0)
        # return the q value if exists else returns 0.0

    def learnQ(self, state,action,reward,value): # the learning procedure for our q function
        '''
        Q learning
        Q(s,a) += alpha*(reward(s,a) + max(Q(s') - Q(s,a)))
        '''
        oldv = self.q.get((state,action),None)
        if oldv is None:
            self.q[(state,action)] = reward
        else:
            self.q[(state,action)] = oldv + self.alpha*(value  - oldv)
    def chooseAction(self,state):
        '''
        Chooses the action according to exxploration and exploitation
        '''
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.getQ(state,a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count >1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
        return action


    def learn(self,state,action,reward,state2):
        maxqnew = max([self.getQ(state2,a) for a in self.actions])
        self.learnQ(state, action, reward, reward + self.gamma*maxqnew)

import math
def ff(f,n):
    fs = "{:f}".format(f)
    if len(fs)<n:
        return ("{:"+n+"s}").format(fs)
    else:
        return fs[:n]
