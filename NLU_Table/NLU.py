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



### Load Data
train_set, valid_set, dicts = data.load.atisfull()
w2idx, ne2idx, labels2idx = dicts['words2idx'], dicts['tables2idx'], dicts['labels2idx']

# Create index to word/label dicts
idx2w  = {w2idx[k]:k for k in w2idx}
idx2ne = {ne2idx[k]:k for k in ne2idx}
idx2la = {labels2idx[k]:k for k in labels2idx}


### Model
n_classes = len(idx2la)
n_vocab = len(idx2w)

model_whole = "./best_model_whole.h5"
model_weights = "best_model_weights.h5"

model = Sequential()
model.add(Embedding(n_vocab,100))
model.add(Convolution1D(64,5,border_mode='same', activation='relu'))
model.add(Dropout(0.25))
model.add(GRU(100,return_sequences=True))
model.add(TimeDistributed(Dense(n_classes, activation='softmax')))
model.compile('rmsprop', 'categorical_crossentropy')

model.load_weights(model_weights)


print("Loaded Model from disk")




#### ###########################333
### model is our model
#####################################



a = raw_input("IC>")
a = a.rstrip()
widx = [w2idx[x] for x in a.split(" ")]
widx = np.array(widx)
widx = widx[np.newaxis, :]
print(a)
print(widx)

pred = model.predict_on_batch(widx)
pred = np.argmax(pred,-1)[0]
print pred
sentence = [idx2la[k] for k in pred]
print a.split(' ')
print sentence



