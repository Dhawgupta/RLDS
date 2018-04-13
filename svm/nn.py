import numpy
import pandas
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
seed = 1337
numpy.random.seed(seed)
import sys,getopt

def main(argv):
    trainFile=''
    validationFile=''
    try:
        opts,argv=getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'Please Provide Name of Training File using -i <FileName>, If you have Validation File also, provide it\'s name Using -o <FileName>'
        sys.exit(2)
        #print 'Bello'
    for opt,arg in opts:
        if opt in("-i","--ifile"):
            trainFile=arg
        elif opt in("-o","--ofile"):
            validationFile=arg
    if trainFile=='':
        print 'ERROR : Please Provide Training File Name'
        sys.exit(-1)
    print 'Will Start Training On File: ',trainFile
    #print 'Validation File: ',validationFile
    TrainNeuralNet(trainFile,validationFile)


def TrainNeuralNet(trainFile,validFile):
    seed = 1337
    numpy.random.seed(seed)
    #Load Training Data
    dataframe = pandas.read_csv(trainFile, header=None,delimiter=' ')
    dataset = dataframe.values
    #number of columns to determine the number of input layer nodes
    cols=dataset.shape[1]
    #Extract all the columns except label column
    X_Train=dataset[:,0:cols-1].astype(float)
    #the label column
    Yytrain=dataset[:,cols-1]
    #one-hot encoding for training labels
    encoder = LabelEncoder()
    encoder.fit(Yytrain)
    encoded_Ytrain = encoder.transform(Yytrain)
    #one-hot encoded vector for label column
    Y_train = np_utils.to_categorical(encoded_Ytrain)
    
    
    #Load Validation Data, same procedure as the training data
    if(validFile!=''):
        print 'Reading from Validation File: ',validFile
        dataframe1 = pandas.read_csv(validFile, header=None,delimiter=' ')
        dataset1 = dataframe1.values
        testcols=dataset1.shape[1]
        X_test=dataset1[:,0:cols-1].astype(float)
        Yy_test=dataset1[:,cols-1]
        encoder.fit(Yy_test)
        encoded_Ytest = encoder.transform(Yy_test)
        Y_test= np_utils.to_categorical(encoded_Ytest)
    else:
        #if no validation file, split the training data, here one hot encoding for labels is done already
        print 'Splitting the dataset in 80-20%'
        X_Train, X_test, Y_train, Y_test = train_test_split(X_Train, Y_train, test_size=0.20, random_state=seed)
    
    print X_test.shape,Y_test.shape,X_Train.shape,Y_train.shape
    
    #prepare the Neural-Net
    batch_size=15000
    #TODO: Is it possible to determine the number of classes automatically ?
    nb_classes=9
    #number of iterations
    nb_epoch=100
    model = Sequential()
    model.add(Dense(4200, input_dim=cols-1))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(4200))
    model.add(Activation('relu'))
    #model.add(D
    
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
              
              
    history = model.fit(X_Train, Y_train,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=1, validation_data=(X_test, Y_test))
    
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig=plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig('Loss.png')

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig=plt.gcf()
    plt.show()
    plt.draw()
    fig.savefig('Accuracy.png')
    
        
if __name__ == "__main__":
    main(sys.argv[1:])            
    

