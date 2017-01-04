import numpy
import sys

num = 10
NUM = num*1000
split = int(NUM*.9)
names = ["feats/benzene%s/0%02d001.npy" % (sys.argv[1], i) for i in xrange(num)]
feats = numpy.vstack([numpy.load(x) for x in names])
#y = numpy.loadtxt("data/benzene_energies.txt")[:NUM] 
y = numpy.loadtxt("data/benzene_energies_ht.txt")[:NUM] * 27

idxs = numpy.arange(NUM)
#numpy.random.shuffle(idxs)
train_idxs = idxs[:split]
test_idxs = idxs[split:]

X = numpy.hstack([feats, numpy.ones((feats.shape[0], 1))])
X_train = X[train_idxs,:]
X_test = X[test_idxs,:]

y_train = y[train_idxs]
y_test = y[test_idxs]

def error(x,y):
    return numpy.abs(x-y).mean()

for alpha in [1e-3, 1e-5, 1e-7, 1e-9]:
    w = numpy.linalg.solve(X_train.T.dot(X_train) + alpha*numpy.eye(X_train.shape[1]), X_train.T.dot(y_train)) 
    print alpha, error(X_train.dot(w), y_train), error(X_test.dot(w), y_test)

import matplotlib.pyplot as plt

mm = numpy.linalg.eig(X.T.dot(X))
print mm[0][:4].real / mm[0].real.sum()
plt.scatter(X.dot(mm[1][:,0]).real, X.dot(mm[1][:, 1]).real, c=y, lw=0)

#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X.dot(mm[1][:,0]).real, X.dot(mm[1][:, 1]).real, X.dot(mm[1][:, 2]).real, c=y, lw=0)

plt.show()


all_forces = numpy.loadtxt("data/benzene_forces_ht.txt")[:NUM, :]
all_forces *= 51.42207
names2 = ["feats/benzeneDereps3/0%02d001.npy" % i for i in xrange(num)]
epsfeats = numpy.vstack([numpy.load(x) for x in names2])
b = numpy.concatenate([epsfeats, numpy.ones((epsfeats.shape[0], epsfeats.shape[1], 1))], axis=2)
res=b.dot(w)
resres = res.reshape(NUM, -1, 2)
pred_force = (-(resres[:, :, 0] - resres[:, :, 1])/ (2*1e-3))
print numpy.abs(all_forces - pred_force).mean(0) / all_forces.std(0)

atom_forces = all_forces.reshape(-1, 12, 3)
atom_fnorm = numpy.linalg.norm(atom_forces, axis=2)

for i in xrange(36):
	plt.subplot(3, 12, i+1)
	plt.scatter(X.dot(mm[1][:,0]).real, X.dot(mm[1][:, 1]).real, c=all_forces[:,i], lw=0)
plt.show()

for i in xrange(12):
	plt.subplot(3, 4, i+1)
	plt.scatter(X.dot(mm[1][:,0]).real, X.dot(mm[1][:, 1]).real, c=atom_fnorm[:,i], lw=0)
plt.show()
