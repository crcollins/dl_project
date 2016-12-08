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
feats1 = numpy.vstack([numpy.load(x) for x in names1])
feats2 = numpy.vstack([numpy.load(x) for x in names2])
feats = numpy.concatenate([feats1, feats2], axis=2)
X = feats.astype(numpy.float32)
#y = numpy.loadtxt("data/benzene_energies.txt")[:NUM]
y = numpy.loadtxt("data/benzene_energies_ht.txt")[:NUM] * 27


X_train, y_train, X_test, y_test = get_test_train(X, y, split)



y_mean = y_train.mean()
y_std = y_train.std()
y_train = (y_train - y_mean) / y_std


atom_input = Input(shape=(X.shape[2], ))
x1 = Dense(100, activation='tanh')(atom_input)
x2 = Dense(50, activation='tanh')(x1)
out = Dense(1)(x2)

atom_model = Model(atom_input, out)

atoms = [Input(shape=(X.shape[2], )) for i in xrange(X.shape[1])]
outs = [atom_model(atom) for atom in atoms]

res = merge(outs, mode='sum')
model = Model(atoms, res)


model.compile(optimizer='adam', loss='mae')
print(model.summary())
pprint.pprint(model.get_config())

model.fit([X_train[:,i,:] for i in xrange(12)], y_train, nb_epoch=1000, validation_split=0.2)
model.save("models/atom_model.keras")
print(model.evaluate([X_train[:,i,:] for i in xrange(12)], y_train)*y_std)
print(model.evaluate([X_test[:,i,:] for i in xrange(12)], (y_test-y_mean)/y_std)*y_std)
