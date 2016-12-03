from __future__ import print_function

import sys
import pprint 

import numpy

from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.normalization import BatchNormalization

from utils import get_test_train


num = 50
NUM = num * 1000
split = int(NUM*0.9)
names = ["feats/benzene%s/0%02d001.npy" % (sys.argv[1], i) for i in xrange(num)]
feats = numpy.vstack([numpy.load(x) for x in names])
X = numpy.hstack([feats, numpy.ones((feats.shape[0], 1))])
#y = numpy.loadtxt("data/benzene_energies.txt")[:NUM]
y = numpy.loadtxt("data/benzene_energies_ht.txt")[:NUM] * 27


X_train, y_train, X_test, y_test = get_test_train(X, y, split)


y_mean = y_train.mean()
y_std = y_train.std()
y_train = (y_train - y_mean) / y_std



model = Sequential([
	Dense(300, activation='relu', init='normal', input_dim=X_train.shape[1]),
	Dense(300, activation='relu', init='normal'),
	Dense(300, activation='relu', init='normal'),
	Dense(300, activation='relu', init='normal'),
	Dense(300, activation='relu', init='normal'),
	Dense(200, activation='relu', init='normal'),
	Dense(200, activation='relu', init='normal'),
	Dense(100, activation='relu', init='normal'),
	Dense(1),
])

model.compile(optimizer='adam', loss='mae')
print(model.summary())
pprint.pprint(model.get_config())

model.fit(X_train, y_train, nb_epoch=1000, validation_split=0.2)
print(model.evaluate(X_train, y_train)*y_std)
print(model.evaluate(X_test, (y_test-y_mean)/y_std)*y_std)
