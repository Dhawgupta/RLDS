"""
Usage :
python test_Policy.py ./save/filename (episodes)
this command will load the model and run for 10 experiments
"""
from DQN import DQNAgent
import numpy as np
import pickle
import sys
import env
#from SVM_test1 import *
import random
import time
import matplotlib.pyplot as plt
from functions import one_hot_encoding

if len(sys.argv) > 1:
    # the filename has been given
    filename = sys.argv[1]  # contains the fielname
    """
    	if 'save' in filename:
        # the file contains the whole path
        pass
    else:
        filename = "./save/" + filename
	"""

else:
    filename = './save/ENVORIGINAL_2018-04-12 00:55:53.653455_[75]_0.0_0.05_0.7_relu_5000.h5'
print("Loading File {}".format(filename))
eps = None
if len(sys.argv) > 2:
    eps = sys.argv[2]  # the number of episodes to run for
time.sleep(1)
d = env.DialougeSimulation()
# state_size = d.state_size
action_size = d.actions
state_size = d.state_size
agent = DQNAgent(state_size, action_size, hiddenLayers=[75], dropout=0.0, activation='relu', loadname=filename,
                 saveIn=False, learningRate=0.05, discountFactor=0.7,
                 epsilon=0.01)  # epislon is zero as we are testing and hence no exploration

# print(agent.model.get_weights())
time.sleep(1)
TotalReward = 0
if eps is not None:
    Episodes = int(eps)
else:
    Episodes = 100
rewards = []
agent.load(filename)
i=0

for e in range(Episodes):
    state = d.reset()
    all_act = []
    for z in range(0, action_size):
        all_act.append(z)
    #print(all_act)
    done = False
    running_reward = 0
    # make the initial state
    l = len(state)
    stateOriginal = state.copy()
    state = np.reshape(state, [l])
    #state = np.append(state, np.zeros(action_size))
    #state = np.append(state, np.zeros(l))

    state = np.reshape(state, [1, len(state)])
    # print("\n\n\n\n")
    print(state)
    # time.sleep(0.5)

    while done == False:
        i = i + 1
        done = False
	#all_act=predict(state)
	print(all_act)
        action = agent.act(state, all_act)
        print "The action taken : %d " % action
        next_state, reward, done = d.step(action)        
        next_stateOriginal = next_state.copy()
        print "the Reward : %f" % reward
        # l = len(next_state)
        next_state = np.reshape(next_state, [l])
        #next_state = np.append(next_state, one_hot_encoding(action_size, action))
        #next_state = np.append(next_state, stateOriginal)
        next_state = np.reshape(next_state, [1, state_size])
        print("next_state==============")
        print(next_state)
        print("i=" + str(i))
    	state = next_state
    	stateOriginal = next_stateOriginal.copy()
    # add the total reward
    	#rewards.append(reward)
	running_reward +=reward
    rewards.append(running_reward)
    TotalReward += running_reward

print("The average Reward is {}".format(TotalReward / Episodes))
print("The average Dialogue length is {}".format(i / Episodes))
print(running_reward)
print(TotalReward)
print(i)
# plt.plot(range(len(rewards)),rewards)
# plt.show()
