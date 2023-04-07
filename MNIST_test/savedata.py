from sklearn.datasets import fetch_openml
import numpy as np

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

np.savetxt("xtrain.csv", X_train, delimiter = ',', fmt='%s')
np.savetxt("xtest.csv", X_test, delimiter = ',', fmt='%s')
np.savetxt("ytrain.csv", y_train, delimiter = ',', fmt='%s')
np.savetxt("ytest.csv", y_test, delimiter = ',', fmt='%s')