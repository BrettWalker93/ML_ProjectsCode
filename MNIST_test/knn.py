import numpy as np

X_train = np.genfromtxt('xtrain.csv', delimiter=',',dtype=float)
X_test = np.genfromtxt('xtest.csv', delimiter=',',dtype=float)
y_train = np.genfromtxt('ytrain.csv', delimiter=',',dtype=float)
y_test = np.genfromtxt('ytest.csv', delimiter=',',dtype=float)

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#kNN
#n_neighbors
#weights = 'uniform', 'distance'
#algorithm = 'brute'
print("kNN Classifier:")
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("default parameters:",metrics.accuracy_score(y_test, y_pred))

clf = KNeighborsClassifier(weights = 'distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("weights = distance:",metrics.accuracy_score(y_test, y_pred))

clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("k = 3:",metrics.accuracy_score(y_test, y_pred))

clf = KNeighborsClassifier(n_neighbors = 3, weights = 'distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("k = 3, weights = distance:",metrics.accuracy_score(y_test, y_pred))

clf = KNeighborsClassifier(n_neighbors = 25)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("k = 25:",metrics.accuracy_score(y_test, y_pred))

clf = KNeighborsClassifier(n_neighbors = 25, weights = 'distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("k = 25, weights = distance:",metrics.accuracy_score(y_test, y_pred))

clf = KNeighborsClassifier(algorithm = 'brute')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("algo = brute:",metrics.accuracy_score(y_test, y_pred))

clf = KNeighborsClassifier(algorithm = 'brute', weights = 'distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("algo = brute, weights = distance:",metrics.accuracy_score(y_test, y_pred))

clf = KNeighborsClassifier(algorithm = 'brute', n_neighbors = 3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("algo = brute, k = 3:",metrics.accuracy_score(y_test, y_pred))

clf = KNeighborsClassifier(algorithm = 'brute', n_neighbors = 3, weights = 'distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("algo = brute, k = 3, weights = distance:",metrics.accuracy_score(y_test, y_pred))


clf = KNeighborsClassifier(algorithm = 'brute', n_neighbors = 25)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("algo = brute, k = 25:",metrics.accuracy_score(y_test, y_pred))

clf = KNeighborsClassifier(algorithm = 'brute', n_neighbors = 25, weights = 'distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("algo = brute, k = 25, weights = distance:",metrics.accuracy_score(y_test, y_pred))
