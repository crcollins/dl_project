import numpy
import sys

num = 10
NUM = num*1000
split = int(NUM*.9)
names = ["feats/benzene%s/0%02d001.npy" % (sys.argv[1], i) for i in xrange(num)]
feats = numpy.vstack([numpy.load(x) for x in names])
y = numpy.loadtxt("data/datasets_public/benzene_energies.txt")[:NUM]

X = numpy.hstack([feats, numpy.ones((feats.shape[0], 1))])
X_train = X[:split,:]
X_test = X[split:,:]
y_train = y[:split]
y_test = y[split:]

def error(x,y):
    return numpy.abs(x-y).mean()

for alpha in [1e-3, 1e-5, 1e-7, 1e-9]:
    w = numpy.linalg.solve(X_train.T.dot(X_train) + alpha*numpy.eye(X_train.shape[1]), X_train.T.dot(y_train)) 
    print alpha, error(X_train.dot(w), y_train), error(X_test.dot(w), y_test)
