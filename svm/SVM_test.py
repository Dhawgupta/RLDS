#from sklearn.datasets import load_svmlight_file
import numpy as np
import csv
import pickle
import heapq
#import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
#X, Y = load_svmlight_file("Train/Train-4-1.txt.svm")


X_new=[]
X_new_preds=[]
Y_new=[]
Y_new_preds=[]



with open('naive.pkl', 'rb') as f:
    clf = pickle.load(f)
    print "classifier loaded"
#X_new_preds = clf.predict(X_new)
with open('test.txt', 'r') as f_train:
     for line in f_train:
        my_list_train = line.split(',') # replace with your own separator instead
	X_new.append(my_list_train)

for i in range(0,len(X_new)):
	for j in range(0,len(X_new[i])):
		X_new[i][j]=float(X_new[i][j])
print(X_new)
X_new_preds = clf.predict(X_new) 
prediction_proba=clf.predict_proba(X_new)
print(prediction_proba)

p_p=[]
for i in range(0,len(prediction_proba)):
	for j in range(0,len(prediction_proba[i])):
		p_p.append(prediction_proba[i][j])
print(p_p)
	
large=heapq.nlargest(4, p_p)
print(large)
for i, j in enumerate(p_p):
	for k in range(0,len(large)):
		if j == large[k]:
			print(i)



'''
p_p=[]
for i in range(0,len(prediction_proba)):
	for j in range(0,len(prediction_proba[i])):
		p_p.append(prediction_proba[i][j])
print(p_p)
	
#large=heapq.nlargest(2, p_p)
#print(large)
for i, j in enumerate(p_p):
		if j >= 0.001:
			print(i)

for pred in X_new_preds:
    if(pred.strip()=='find_flight'):
        #load the feature vec and find_flight model
        with open('find_flight.pkl', 'rb') as f:
        	clf = pickle.load(f)
        	print "classifier loaded"
        with open('with.arff', 'r') as f_train:
        	for line in f_train:
                	my_list_train = line.split(',') # replace with your own separator instead
	        	Y_new.append(my_list_train)
        Y_new_preds = clf.predict(Y_new) 
    elif(pred.strip()=='service'):
        #same
        with open('service.pkl', 'rb') as f:
        	clf = pickle.load(f)
        	print "classifier loaded"
        with open('without.arff', 'r') as f_train:
        	for line in f_train:
                	my_list_train = line.split(',') # replace with your own separator instead
	        	Y_new.append(my_list_train)
        Y_new_preds = clf.predict(Y_new) 

    elif(pred.strip()=='airline'):
        with open('airline.pkl', 'rb') as f:
        	clf = pickle.load(f)
        	print "classifier loaded"
        with open('without.arff', 'r') as f_train:
        	for line in f_train:
                	my_list_train = line.split(',') # replace with your own separator instead
	        	Y_new.append(my_list_train)
        Y_new_preds = clf.predict(Y_new) 

        
print"testing completed"
#print Y_test
for pred in X_new_preds:
	print(pred.strip())
for pred in Y_new_preds:
	print(pred.strip())
'''
print "testing completed"
