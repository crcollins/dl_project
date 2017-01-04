from __future__ import print_function

import sys
import pprint 

import numpy

from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import Input, merge, Lambda, Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.pooling import MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

from utils import get_test_train

num = 1
NUM = num * 1000
EPS = 1e-3
split = int(NUM*0.9)
names1 = ["feats/benzenelocalencDereps3/0%02d001.npy" % i for i in xrange(num)]
names2 = ["feats/benzeneatom/0%02d001.npy" % i for i in xrange(num)]
feats1 = numpy.vstack([numpy.load(x) for x in names1])
feats2 = numpy.vstack([numpy.load(x) for x in names2])
new = numpy.tile(feats2[:, None, :, :], (1, 72, 1, 1))
feats = numpy.concatenate([feats1, new], axis=3)
y = numpy.loadtxt("data/benzene_forces_ht.txt")[:NUM, :] * 51.42207

X = feats.astype(numpy.float32)
'''
for x1 in X:
    for x2 in x1:
	    numpy.random.shuffle(x2)
'''

X_train, y_train, X_test, y_test = get_test_train(X, y, split)


y_mean = y_train.mean(0)
y_std = y_train.std(0)
y_train = (y_train - y_mean) / y_std

E_model = load_model("models/lstm_tanh.keras")

pt1 = Input(shape=X.shape[2:])
pt2 = Input(shape=X.shape[2:])
Ept1 = E_model(pt1)
Ept2 = E_model(pt2)

out = merge([Ept1, Ept2], mode=lambda x: (x[0] - x[1]) / (-2 * EPS), output_shape=(1, ))

F_model = Model([pt1, pt2], out)


mol_input = Input(shape=X.shape[1:])

def select(x, i):
	return x[:, i]
selected = [[Lambda(select, arguments={'i': i})(mol_input), Lambda(select, arguments={'i': i+1})(mol_input)] for i in xrange(0, X.shape[1], 2)]
forces = [F_model(x) for x in selected]
final = merge(forces, mode='concat') 


model = Model(mol_input, final)
#model.load_weights("/tmp/lstm_force_convert_weights.hdf5")
model.compile(optimizer='adam', loss='mae')
print(model.summary())
pprint.pprint(model.get_config())

checkpointer = ModelCheckpoint(filepath="/tmp/lstm_force_convert_weights.hdf5", verbose=1)

model.fit(X_train, y_train, nb_epoch=1000, validation_split=0.2, callbacks=[checkpointer])
'''n_epochs = 1000
for i in xrange(n_epochs):
	for X_batch, y_batch in batches:
		model.fit(X_batch, y_batch, nb_epoch=1, validation_split=0.2, callbacks=[checkpointer])
'''
model.save("models/lstm_force_convert.keras")

print(model.evaluate(X_train, y_train)*y_std)
print(model.evaluate(X_test, (y_test-y_mean)/y_std)*y_std)


