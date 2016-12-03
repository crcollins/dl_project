from __future__ import print_function

import sys
import pprint 

import numpy

from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, merge 

from utils import get_test_train


num = 50
NUM = num * 1000
split = int(NUM*0.9)
names1 = ["feats/benzenelocalenc/0%02d001.npy" % i for i in xrange(num)]
names2 = ["feats/benzeneatom/0%02d001.npy" % i for i in xrange(num)]
names3 = ["feats/benzeneBP/0%02d001.npy" % i for i in xrange(num)]
feats1 = numpy.vstack([numpy.load(x) for x in names1])
feats2 = numpy.vstack([numpy.load(x) for x in names2])
feats3 = numpy.vstack([numpy.load(x) for x in names3])
feats = numpy.concatenate([feats1, feats2, feats3], axis=2)
y = numpy.loadtxt("data/benzene_energies.txt")[:NUM]


X_train, y_train, X_test, y_test = get_test_train(X, y, split)


y_mean = y_train.mean()
y_std = y_train.std()
y_train = (y_train - y_mean) / y_std


atom_input = Input(shape=(X.shape[2], ))
#x1 = Dense(100, activation='tanh')(atom_input)
x2 = Dense(50, activation='tanh')(atom_input)
out = Dense(1)(x2)

atom_model = Model(atom_input, out)

atoms = [Input(shape=(X.shape[2], )) for i in xrange(X.shape[1])]
outs = [atom_model(atom) for atom in atoms]

res = merge(outs, mode='sum')
model = Model(atoms, res)

def loss(y_true, y_pred):
    return K.mean(K.abs(y_true-K.round(y_pred, 1)))

model.compile(optimizer='adam', loss='mae')
print(model.summary())
pprint.pprint(model.get_config())

model.fit([X_train[:,i,:] for i in xrange(12)], y_train, nb_epoch=1000, validation_split=0.2)
print(model.evaluate(X_train, y_train)*y_std)
print(model.evaluate(X_test, (y_test-y_mean)/y_std)*y_std)
