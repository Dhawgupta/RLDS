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
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
#X, Y = load_svmlight_file("Train/Train-4-1.txt.svm")
X = []
Y= []
with open('sample.txt', 'r') as f_train:
     for line in f_train:
	my_list_train = line.split(',') # replace with your own separator instead
	print(my_list_train)
	X.append(my_list_train[:-1]) # omitting identifier in [0] and target in [-1]
        
       	Y.append(my_list_train[-1])
       	        #X.append(my_list_train[:-1]) 
       	        #Y.append('RAT')
       	        
print X
print Y
for i in range(0,len(X)):
	for j in range(0,len(X[i])):
		X[i][j]=float(X[i][j])
Y = map(int, Y)
print X
print Y

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=0)
#rfc=GaussianNB()
rfc= SVC(C=1.0, kernel='linear',probability=True)
rfc.fit(X_train, Y_train) 
prediction_proba=rfc.predict_proba(X_test)
prediction_label=rfc.predict(X_test)
print(prediction_proba)
'''
p_p=[]
for i in range(0,len(prediction_proba)):
	for j in range(0,len(prediction_proba[i])):
		p_p.append(prediction_proba[i][j])
print(p_p)
	
large=heapq.nlargest(2, p_p)
print(large)
for i, j in enumerate(p_p):
	for k in range(0,len(large)):
		if j == large[k]:
			print(i)

res=zip(prediction_label,prediction_proba)
with open('Result.result', 'w') as f:
       writer = csv.writer(f, delimiter=' ')
       writer.writerows(zip(prediction_proba))
print(accuracy_score(Y_test, prediction_label))
print(classification_report(Y_test, prediction_label,digits=4))
print(confusion_matrix(Y_test, prediction_label))
'''
with open('naive.pkl', 'wb') as f:
    pickle.dump(rfc, f)
    print "classifier saved"

print(X_test)

