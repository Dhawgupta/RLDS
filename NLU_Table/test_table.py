"""
Usage : ussed to test the table function
./save/filename


"""

import numpy as np
import sys
import random
import time
from functions import one_hot_encoding, convertState
import pickle 
import json
import env
import time
from qlearn_table import Qlearn
import matplotlib.pyplot as plt




if len(sys.argv) > 1:
	filename  = sys.argv[1] 
	if 'save' in filename:
		pass # the path is whole
	else:
		filename = "./save/" + filename

else:
	filename = './save/Table_dialouge_2018-01-24 15:08:05.385535_0.1_0.9_0.2_100000.pkl'

print("The loading File {}".format(filename))
time.sleep(1)

# Filename
# selection for pkl file or json file
data = dict()
if 'json' in filename:
	json_file = open(filename)
	data = json.load(json_file)

elif 'pkl' in filename:
	with open(filename,'r') as handle:
		data = pickle.load(handle)


# print(data)
d = env	.DialougeSimulation() # create a object of the env
Episodes = 10
if len(sys.argv) > 2:
	# includes the number of epusodes
	Episodes = int(sys.argv[2])



total_reward = 0
state_size = d.state_size
# print "The Size of State is : ",state_size
action_size =  d.actions
# keeping the epsilon to be zero so that no exploration
agent = Qlearn([i for i in range(action_size)], epsilon = 0, alpha = 0.2, gamma = 0.9)
# set the agent q table to be the q table that is loaded
agent.q = data.copy() # copied the dictionary from data to q
rewards = []
for e in range(Episodes):
	state = d.reset()
	done = False
	running_reward = 0
	state = np.reshape(state, [state_size, ])
	i = 0
	while done == False:
		i+=1
		action = agent.chooseAction(convertState(state))

		next_state, reward, done = d.step(action)
		next_state = np.reshape(next_state, [state_size,])
		running_reward += reward
		print("State: {} , action : {}".format(state, action))
		if done:
			print ("Episode : {}/{}, Reward : {} , Steps : {}".format(e,Episodes, running_reward, i))
			print ("State is {}".format(state))
			total_reward += running_reward
			rewards.append(running_reward)

			break
		state = next_state


print("The average Reward is {}".format(total_reward/Episodes))
plt.plot(range(len(rewards)	),rewards)
plt.show()
#plt.savefig('./figures/' + filename.split('/')[2] + '.png')
	

