import numpy as np
import pickle

import data.load
# from metrics.accuracy import conlleval

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution1D, MaxPooling1D
import progressbar

#print("Loaded Model from disk")

class NLU:
    def __init__(self,filename = "best_model_weights.h5",load = True):
        # loading the data and dictioanries
        self.train_set, self.valid_set, self.dicts = data.load.atisfull()
        self.w2idx, self.ne2idx, self.labels2idx = self.dicts['words2idx'], self.dicts['tables2idx'], self.dicts['labels2idx']

        self.idx2w = {self.w2idx[k]: k for k in self.w2idx}
        self.idx2ne = {self.ne2idx[k]: k for k in self.ne2idx}
        self.idx2la = {self.labels2idx[k]: k for k in self.labels2idx}

        self.n_classes = len(self.idx2la)
        self.n_vocab = len(self.idx2w)
        self.model = self.create_model()
        self.filename = filename
        if load:
            self.load_model_weights(filename)

    def load_model_weights(self,filename):
        print ("Loading the NLU model")
        self.model.load_weights(filename)

    def create_model(self):
        model = Sequential()
        model.add(Embedding(self.n_vocab, 100))
        model.add(Convolution1D(64, 5, border_mode='same', activation='relu'))
        model.add(Dropout(0.25))
        model.add(GRU(100, return_sequences=True))
        model.add(TimeDistributed(Dense(self.n_classes, activation='softmax')))
        model.compile('rmsprop', 'categorical_crossentropy')
        return model



    def parse_sentence(self,string):
        """
        This will return the various labels for the string
        :param string: The string which is enteree by the user
        :return: return s the labels for each word
        """
        string = string.lower()
        widx = [self.w2idx[x] for x in string.split(" ")]
        widx = np.array(widx)
        widx = widx[np.newaxis, :]
        pred = self.model.predict_on_batch(widx)
        pred1 = np.argmax(pred, -1)[0]
        print pred1
        # sentence contains the corresponding prediceted labels
        sentence = [self.idx2la[k] for k in pred1]
        print string.split(' ')
        print sentence
        pred2 = np.argmax(pred, -1)[0]
        print pred2
        print sum(pred[0][0])
        prob_values = [pred[0][i][pred2[i]] for i in range(len(pred2))]
        print prob_values

        return [sentence, prob_values]
        # widx contains the

