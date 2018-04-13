# we will implement the new state action state tuple in the training
# format
# current_state, previous action, prevuis state
from DQN import DQNAgent
import env
from SVM_test import *
import numpy as np
from datetime import datetime
from functions import one_hot_encoding
from time import sleep
a = str(datetime.now())
Episodes = 200000
d = env.DialougeSimulation()
filename = "./save/ENVdialouge_DDQN"  + a  + "_"
# filename2 =
state_size = d.state_size
# print "The Size of State is : ",state_size
action_size =  d.actions
# state size will be 2states + 1 action space

#state_size = state_size+action_size+state_size

# made dropout to 0
agent = DQNAgent(state_size, action_size, hiddenLayers = [75], dropout = 0.0000, activation = 'relu', loadname = None, saveIn = False, learningRate = 0.05, discountFactor = 0.7)
u = "_"
filename = filename + str(agent.hiddenLayers) + u + str(agent.dropout)+ u +str(object=agent.learning_rate) + u + str(object=agent.gamma) + u + agent.activation + u + str(object=Episodes)+ ".h5"
done = False
batch_size = 32
i =0
track=[]
for e in range(Episodes):
	n=1
	state = d.reset()
	all_act=[]
	#for z in range(0,action_size):
	#	all_act.append(z)
	#print(all_act)
	done = False
	running_reward = 0
	#make the initial state
	l = len(state)
	stateOriginal = state.copy()
	state = np.reshape(state , [l])
	#print(state)
	#state = np.append(state,np.zeros(action_size))
	#state = np.append(state,np.zeros(l))
	state = np.reshape(state, [1, len(state)])
	print(state)
	# print("\n\n\n\n")
	#print(state.shape)
	# time.sleep(0.5)

	while done == False:
		done = False
		all_act=predict(state)
		print(all_act)
		action = agent.act(state,all_act)
		#all_act.remove(action)
		#print(all_act)
		print "The action taken : %d "%action
		next_state, reward, done = d.step(action)
		next_stateOriginal = next_state.copy()
		print "the Reward : %f" %reward
		# l = len(next_state)
		next_state = np.reshape(next_state , [l])
		#next_state = np.append(next_state, one_hot_encoding(action_size,action))
		#next_state = np.append(next_state, stateOriginal)
		next_state = np.reshape(next_state, [1, state_size])
		print(next_state)



		agent.remember(state, action, reward, next_state, done)
		i  = i+1

		if len(agent.memory) > batch_size:
			agent.replay(batch_size)
		if i%agent.update_freq == 0:
			agent.updateTargetModel()
		agent.rem_rew(reward)
		running_reward = running_reward + reward
		print("e="+str(e))
		print("i="+str(i))
		# print("State : {}\nAction : {}\nNextState : {}\n".format(state,action,next_state))
		if i % 100 == 0:            # calculating different variables to be outputted after every 100 time steps
			avr_rew = agent.avg_rew()
			track.append([str(i)+" "+str(avr_rew)+" "+str(e)+" "+str(agent.epsilon)])
			with open("results1" + a + "_.txt", 'w') as fi:
				for j in range(0,len(track)):
					line=track[j]
					fi.write(str(line).strip("[]''")+"\n")
		#print(track)
		

		if done:
			print("episode: {}/{}, score: {}, e: {:.2}"
				  .format(e, Episodes, running_reward, agent.epsilon))
			print("The state is : ",next_state)
			print "\n\n"
			break
		#i  = i+1
		stateOriginal = next_stateOriginal.copy()
		state =next_state
		# time.sleep(1)

		# if done:
		# 	print "episode: {}/{}, score: {}, e: {:.2}".format(e, Episodes, running_reward, agent.epsilon)
	#        	break

	# =i+1
			# print("e="+str(e))
			# state = next_state
	#if len(agent.memory) > batch_size:
		#agent.replay(batch_size)

	if e % 200 == 0:
		print "Saving : {}".format(e)
		agent.save(filename)
		sleep(0.2)
		print "Done saving now we can quit"
		sleep(1)
		# print(agent.model.get_weights())
