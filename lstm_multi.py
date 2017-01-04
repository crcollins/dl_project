from __future__ import print_function

import sys
import pprint 

import numpy

from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation, Flatten, Dropout
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
y = numpy.loadtxt("data/benzene_props.txt")[:NUM, :]

X = feats.astype(numpy.float32)
for x in X:
    numpy.random.shuffle(x)


X_train, y_train, X_test, y_test = get_test_train(X, y, split)


y_mean = y_train.mean(0)
y_std = y_train.std(0)
y_train = (y_train - y_mean) / y_std


input_ = Input(shape=X.shape[1:])
lstm = LSTM(32, input_shape=(12, X_train.shape[2]))(input_)
layer = Dense(100, activation='relu', init='normal')(lstm)
outs = [Dense(1)(layer) for i in xrange(y.shape[1])]

model = Model(input_, outs)
model.compile(optimizer='adam', loss='mae')
print(model.summary())
pprint.pprint(model.get_config())

checkpointer = ModelCheckpoint(filepath="/tmp/lstm_multi_weights.hdf5", verbose=1)
model.fit(X_train, [y_train[:, i] for i in xrange(y.shape[1])], nb_epoch=1000, validation_split=0.2, callbacks=[checkpointer])
model.save("models/lstm_multi.keras")
print(model.evaluate(X_train, y_train)*y_std)
print(model.evaluate(X_test, (y_test-y_mean)/y_std)*y_std)
