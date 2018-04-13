import numpy as np
import random


def checkState(state):
	threshold = 0.70
	if ( state[1]  >= threshold and state[2]  >= threshold and state[3]  >= threshold and state[4]  >= threshold and  state[0]  >= threshold ):
		
		return True
	else:
		return False

# convert [0.324,0.45] -> [0.30,0.4]return [3,4] make it integer so that easy to work with in  dictionary for lookup tables

def convertState(state):# round the state to the nearest value
	# print state
	# TODO change for resolution 0.1 => round(i,1)*10 , 0.01 => round(i,2)*100s
	newstate = [int(round(i,2)*100) for i in state ] # increasing the resloution form 0.1 etc to 0.01
	return tuple(newstate)
def convertState10(state):# round the state to the nearest value
	# print state
	# TODO change for resolution 0.1 => round(i,1)*10 , 0.01 => round(i,2)*100s
	newstate = [int(round(i,1)*10) for i in state ] # increasing the resloution form 0.1 etc to 0.01
	return tuple(newstate)

def convertState100(state):# round the state to the nearest value
	# print state
	# TODO change for resolution 0.1 => round(i,1)*10 , 0.01 => round(i,2)*100s
	newstate = [int(round(i,2)*100) for i in state ] # increasing the resloution form 0.1 etc to 0.01
	return tuple(newstate)
def one_hot_encoding(size, index):
	a = np.zeros([size])
	a[index] = 1
	return a
def soft_check_state(state):
	"""
	returns the factor to be multipled to the total reward
	3 states => 1/3
	4 states => 2/3
	5 states => 1
	else -1
	"""
	threshold = 0.70
	count =0
	for i in range(1,6):
		if state[i] > threshold:
			count+=1

	if count>2:
		return (count -2)/3.0
	else:
		return -1
