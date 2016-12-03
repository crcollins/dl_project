from __future__ import print_function

import sys
import pprint 

import numpy

from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, merge 

num = 5
NUM = num * 1000
split = int(NUM*0.9)
names1 = ["feats/benzenelocalenc2/0%02d001.npy" % i for i in xrange(num)]
names2 = ["feats/benzeneatom/0%02d001.npy" % i for i in xrange(num)]
feats1 = numpy.vstack([numpy.load(x) for x in names1])
feats2 = numpy.vstack([numpy.load(x) for x in names2])
feats = numpy.concatenate([feats1, feats2], axis=2)
y = numpy.loadtxt("data/benzene_energies.txt")[:NUM]
y2 = numpy.loadtxt("data/benzene_forces.txt")[:NUM, :] 
y2 = numpy.linalg.norm(y2.reshape(-1, 12, 3), axis=2)


X = feats.astype(numpy.float32)
y = y.astype(numpy.float32)
y2 = y2.astype(numpy.float32)
X_train = X[:split, :]
y_train = y[:split]
y2_train = y2[:split,:]
X_test = X[split:, :]
y_test = y[split:]
y2_test = y2[split:, :]


y_mean = y_train.mean()
y_std = y_train.std()
y_train = (y_train - y_mean) / y_std

y2_mean = y2_train.mean(0)
y2_std = y2_train.std(0)
y2_train = (y2_train - y2_mean) / y2_std



atom_input = Input(shape=(X.shape[2], ))
x1 = Dense(100, activation='relu')(atom_input)
x2 = Dense(50, activation='relu')(x1)
out = Dense(1)(x2)
out2 = Dense(1)(x2)

atom_model = Model(atom_input, [out, out2])

atoms = [Input(shape=(X.shape[2], )) for i in xrange(X.shape[1])]
outs = [atom_model(atom) for atom in atoms]

res = merge([x[0] for x in outs], mode='sum')
res2 = merge([x[1] for x in outs], mode='concat')
model = Model(atoms, output=[res, res2])


model.compile(optimizer='adam', loss='mae')#, loss_weights=[1., 0.])
print(model.summary())
pprint.pprint(model.get_config())

model.fit([X_train[:,i,:] for i in xrange(12)], [y_train, y2_train], nb_epoch=1000, validation_split=0.2)
print(model.evaluate(X_train, y_train)*y_std)
print(model.evaluate(X_test, (y_test-y_mean)/y_std)*y_std)
