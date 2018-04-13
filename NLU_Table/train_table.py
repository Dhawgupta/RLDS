from qlearn_table import Qlearn
import env
import numpy as np
from datetime import datetime
from functions import *
import json
import pickle

a = str(datetime.now())
Episodes = 10000000

d = env.DialougeSimulation()
# TODO change the resolution accordingly and random
filename = "./save/Table_5_slot_dialouge_with_resoultion_0.01_with_randomstate_"  + a  + "_"
filename_reward = "./save/Rewards_resolution_0.1_with_randomstate_" + a  + "_Table.pkl" # file to store the
# TODO change resolution accrodingy and random
results_file = "results_table_resolution0.01_with_randomstate_" + a+"_.txt"# rewards
# filename2 =
state_size = d.state_size
# print "The Size of State is : ",state_size
action_size =  d.actions
agent = Qlearn([i for i in range(action_size)], epsilon = 0.1, alpha = 0.2, gamma = 0.9)
u = "_"
filename = filename + "_Updated_environment_" +str(object=agent.epsilon) + u + str(object=agent.gamma) + u + str(agent.alpha) + u + str(object=Episodes)+  ".pkl"
done = False
track = []
for e in range(Episodes):
	state = d.reset()
	done = False
	# running_reward = 0
	state = np.reshape(state, [state_size,])
	running_reward =0
	reward_for_episodes = []
	while done == False:
		# done = False
		action = agent.chooseAction(convertState(state))
		# print "The action taken : %d "%action
		next_state, reward, done = d.step(action)
		# print "the Reward : %f" %reward
		next_state = np.reshape(next_state, [state_size,])
		# agent.remember(state, action, reward, next_state, done)
		agent.learn(convertState(state), action, reward, convertState(next_state))
		running_reward = running_reward + reward
		agent.rem_rew(reward)
		i = i + 1
		if i%100 == 0:
			avr_rew = agent.avg_rew()
			track.append([str(i) + " " + str(avr_rew) + " " + str(e) + " " + str(agent.epsilon)])
			with open(results_file,'w') as fl:
				for j in range(0,len(track)):
					line = track[j]
					fl.write(str(line).strip("[]''") + "\n")
		if done:
			print("episode: {}/{}, score: {}, e: {:.2}"
	              .format(e, Episodes, running_reward, agent.epsilon))
			print("The state is : ",state)
			print "\n\n"
			break

		state = next_state
		# if done	:
		# 	print "episode: {}/{}, score: {}, e: {:.2}".format(e, Episodes, running_reward, agent.epsilon)
	#        	break

		reward_for_episodes.append(running_reward)
		
	# =i+1
	    	# print("e="+str(e))
	    	# state = next_state
	# if len(agent.memory) > batch_size:
	#     agent.replay(batch_size)


	if e%100 == 0:
		with open(filename,'wb') as f:
			print "Saving to file"
			pickle.dump(agent.q, f, protocol=pickle.HIGHEST_PROTOCOL)


