import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation, Dropout
from keras.optimizers import Adam
from keras import optimizers
from keras.models import load_model
import time

EPISODES = 10


class DQNAgent:
    def __init__(self, state_size, action_size, hiddenLayers=[], dropout=0.1, activation='relu', loadname=None,
                 saveIn=False, learningRate=0.01, discountFactor=0.9,
                 epsilon=None):  # the file to load is provided if requirede
        # saveIn is providded if to store the model in from which we load it
        print("in init")
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
	self.memory1 = deque(maxlen=100)
        self.gamma = discountFactor  # discount rate
        if epsilon is None:
            self.epsilon = 1.0  # exploration rate
        else:
            self.epsilon = epsilon
        self.epsilon_min = 0.15 # changed the epsilon from 0.01 to 0.2
        self.epsilon_decay = 0.00005
        self.learning_rate = learningRate
        # self.noHiddenLayers = noHiddenLayers

        self.hiddenLayers = hiddenLayers
        self.dropout = dropout
        self.activation = activation
        self.model = self._build_model(self.hiddenLayers, self.dropout, self.activation)
        self.loadname = loadname
        self.saveInLoaded = saveIn
        self.iter = 0  # the amount of model runs and iterations and runs
        if self.loadname is not None:
            self.load(self.loadname)

    def init_record(self):
        pass
        # f = open(self.filename,'w')
        # f.write('The headers for records separated by commma')
        # f.write('The initial state')
        # f.close()

    def update_record(self):
        # f = open(self.filename, 'a')
        # f.write('write the current state')
        # f.close()
        pass

    def _build_model(self, hiddenLayers, dropout, activation):
        # Neural Net for Deep-Q learning Model
        print("in build_model")
        bias = True
        model = Sequential()
        if len(hiddenLayers) == 0:
            model.add(Dense(self.action_size, inputs_shape=(self.state_size,), kernel_initializer='lecun_uniform',
                            use_bias=bias))
            model.add(Activation("linear"))
        else:
            model.add(Dense(hiddenLayers[0], input_shape=(self.state_size,), kernel_initializer="lecun_uniform",
                            use_bias=bias))
            model.add(Activation(activation))

        for index in range(1, len(hiddenLayers)):
            layerSize = hiddenLayers[index]
            model.add(Dense(layerSize, kernel_initializer="lecun_uniform", use_bias=bias))
            model.add(Activation(activation))
            if dropout > 0:
                model.add(Dropout(dropout))
        model.add(Dense(self.action_size, kernel_initializer="lecun_uniform", use_bias=bias))
        model.add(Activation("linear"))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        optimizer = optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        # print("in remember")
        self.memory.append((state, action, reward, next_state, done))
    def avg_rew(self):
	sum1=0
	i=0
	for elem in self.memory1:
		i=i+1
		sum1=sum1+elem
	avr=sum1/i
        #print(i)
        #print(avr)
	return avr

    def rem_rew(self, reward):
        self.memory1.append((reward))

    def act(self, state, all_act):
        # print("in act")
        if np.random.rand() <= self.epsilon:
            return random.choice(all_act)
        act_values = self.model.predict(state)
        max_key = all_act[0]
        max_val = act_values[0, max_key]
        l = len(all_act)
        for i in range(1, l):
            # print(i)
            k = all_act[i]
            # print(k)
            val_n = act_values[0, k]
            if (val_n > max_val):
                max_val = val_n
                max_key = k
        # print(max_key)
        return max_key  # returns action

    def replay(self, batch_size):
        # print("in replay")
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            # print("state")
            # print(state)
            target_f = self.model.predict(state)
            # print("target_f")
            # print(target_f)
            target_f[0][action] = target
            # print("new target")
            # print(target_f)
            self.model.fit(state, target_f, nb_epoch=1, verbose=0)
        self.iter += 1  # to test the model iteration as well

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def load(self, name):
        print("in load")
        print("Loading the model {} ".format(name))
        self.model = load_model(name)
        # time.sleep(3)
        # self.model.load_weights(name)

    def save(self, name):
        # saveIn tells us
        print("saving in .... {}".format(name))
        if (self.loadname is None) or (self.saveInLoaded is False):
            print "Saving in without loading : {}".format(name)
            # self.model.save_weights(name)
            self.model.save(name)
        elif self.saveInLoaded is True and self.loadname is not None:
            print "Saving in : {}".format(self.loadname)
            # self.model.save_weights(self.loadname)
            self.model.save(self.loadname)
        else:
            print("Error in saving no Conition mathcing")
