import numpy


def get_test_train(X, y, split, shuffle=False):
	X = X.astype(numpy.float32)
	y = y.astype(numpy.float32)
	idxs = numpy.arange(X.shape[0])
	if shuffle:
		numpy.random.shuffle(idxs)
	train_idxs = idxs[:split]
	test_idxs = idxs[split:]
	X_train = X[train_idxs, :]
	y_train = y[train_idxs]
	X_test = X[test_idxs, :]
	y_test = y[test_idxs]
	return X_train, y_train, X_test, y_test

