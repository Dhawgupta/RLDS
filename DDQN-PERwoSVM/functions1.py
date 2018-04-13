import numpy as np
import random
# updating it for state_size = 5

def checkState(state):
	threshold = 0.70
	if ( state[0]  >= threshold and state[1]  >= threshold and state[2]  >= threshold and state[3]  >= threshold and  state[4]  >= threshold ):
		
		return True
	else:
		return False

# convert [0.324,0.45] -> [0.30,0.4]return [3,4] make it integer so that easy to work with in  dictionary for lookup tables

def convertState(state):# round the state to the nearest value
	print state
	newstate = [int(round(i,1)*10) for i in state ]
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
	for i in range(5):
		if state[i] > threshold:
			count+=1

	if count>2:
		return (count -2)/3.0
	else:
		return -1


def new_check_state(state):
	"""
	returns the factor to be multipled to the total reward
	3 states => 1/3
	4 states => 2/3
	5 states => 1
	else -1
	"""
	threshold = 0.70
	count =0
	for i in range(5):
		if state[i] > threshold:
			count+=1

	return count/5.0
