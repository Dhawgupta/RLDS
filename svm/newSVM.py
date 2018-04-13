#from sklearn.datasets import load_svmlight_file
import numpy as np
import csv
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
train_text = []
train_label= []
valid_text=[]
valid_label=[]
with open('unigramtrain.arff', 'r') as f_train:
     for line in f_train:
        my_list_train = line.split(',') # replace with your own separator instead
        if len(my_list_train) > 91:
	        train_text.append(my_list_train[:-1]) # omitting identifier in [0] and target in [-1]
        	#print str(line)
        	train_label.append(my_list_train[-1])
       	        #X.append(my_list_train[:-1]) 
       	        #Y.append('RAT')
       	        
with open('unigramtest.arff', 'r') as t_train:
     for line in t_train:
        my_list_train = line.split(',') # replace with your own separator instead
        if len(my_list_train) > 91:
	        valid_text.append(my_list_train[:-1]) # omitting identifier in [0] and target in [-1]
        	#print str(line)
        	valid_label.append(my_list_train[-1])
       	        #X.append(my_list_train[:-1]) 
       	        #Y.append('RAT')
       	        
#print X
#print Y
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
rfc= SVC(C=1.0, kernel='linear',probability=True)
rfc.fit(train_text, train_label) 
prediction_proba=rfc.predict_proba(valid_text)
prediction_label=rfc.predict(valid_text)
res=zip(prediction_label,prediction_proba)
with open('Result.result', 'w') as f:
       writer = csv.writer(f, delimiter=' ')
       writer.writerows(zip(prediction_label,prediction_proba))
print(accuracy_score(valid_label, prediction_label))
print(classification_report(valid_label, prediction_label,digits=4))
print(confusion_matrix(valid_label, prediction_label))

