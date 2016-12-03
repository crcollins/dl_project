from __future__ import print_function

import sys
import pprint 

import numpy

from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.pooling import MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

from utils import get_test_train


num = 50
NUM = num * 1000
split = int(NUM*0.9)
names1 = ["feats/benzenelocalenc/0%02d001.npy" % i for i in xrange(num)]
names2 = ["feats/benzeneatom/0%02d001.npy" % i for i in xrange(num)]
feats1 = numpy.vstack([numpy.load(x) for x in names1])
feats2 = numpy.vstack([numpy.load(x) for x in names2])
feats = numpy.concatenate([feats1, feats2], axis=2)
#y = numpy.loadtxt("data/benzene_energies.txt")[:NUM]
y = numpy.loadtxt("data/benzene_energies_ht.txt")[:NUM] * 27

X = feats.astype(numpy.float32)
# Random shuffle atom order
for x in X:
    numpy.random.shuffle(x)


X_train, y_train, X_test, y_test = get_test_train(X, y, split)


y_mean = y_train.mean()
y_std = y_train.std()
y_train = (y_train - y_mean) / y_std


model = Sequential([
	#GRU(32, input_shape=(12, X_train.shape[2])),
	LSTM(32, input_shape=(12, X_train.shape[2])),
	#Dense(200, activation='relu', init='normal'),
	Dense(100, activation='tanh', init='normal'),
	Dense(1),
])

model.compile(optimizer='adam', loss='mae')
print(model.summary())
pprint.pprint(model.get_config())


#checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1)
model.fit(X_train, y_train, nb_epoch=1000, validation_split=0.2)#, callbacks=[checkpointer])
model.save("models/lstm_tanh.keras")
print(model.evaluate(X_train, y_train)*y_std)
print(model.evaluate(X_test, (y_test-y_mean)/y_std)*y_std)
