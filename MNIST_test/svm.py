import numpy as np

X_train = np.genfromtxt('xtrain.csv', delimiter=',',dtype=float)
X_test = np.genfromtxt('xtest.csv', delimiter=',',dtype=float)
y_train = np.genfromtxt('ytrain.csv', delimiter=',',dtype=float)
y_test = np.genfromtxt('ytest.csv', delimiter=',',dtype=float)

from sklearn import svm, metrics
print("SVM:")
#linear kernel
print("Linear:")
#c = 1.0
clf = svm.SVC(kernel = 'linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("c = 1.0, :",metrics.accuracy_score(y_test, y_pred))

#c = 100.0
clf = svm.SVC(kernel = 'linear',  C=100.0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("c = 100.0:",metrics.accuracy_score(y_test, y_pred))

#c = 0.01
clf = svm.SVC(kernel = 'linear', C=0.01)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("c = 0.1:",metrics.accuracy_score(y_test, y_pred))

#rbf kernel
print("rbf:")
#c = 1.0
clf = svm.SVC(kernel = 'rbf')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("c = 1.0, gamma = default:",metrics.accuracy_score(y_test, y_pred))

#
clf = svm.SVC(kernel = 'rbf', gamma = (1/ 5000))
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("c = 1.0, gamma = 1 / 5000:",metrics.accuracy_score(y_test, y_pred))

#c = 3.0
clf = svm.SVC(kernel = 'rbf', C = 3.0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("c = 3.0, gamma = default:",metrics.accuracy_score(y_test, y_pred))

#
clf = svm.SVC(kernel = 'rbf', gamma = (1/ 5000), C = 3.0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("c = 3.0, gamma = 1 / 5000:",metrics.accuracy_score(y_test, y_pred))

#
clf = svm.SVC(kernel = 'rbf', gamma = (1/ 150), C = 3.0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("c = 3.0, gamma = 1 / 150:",metrics.accuracy_score(y_test, y_pred))

#c = 15.0
clf = svm.SVC(kernel = 'rbf', C = 15.0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("c = 15.0, gamma = default:",metrics.accuracy_score(y_test, y_pred))

#polynomial kernel
print("Polynomial:")
#c = 1.0
clf = svm.SVC(kernel = 'poly')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("c = 1.0, gamma = default:",metrics.accuracy_score(y_test, y_pred))

#c = 1.0
clf = svm.SVC(kernel = 'poly')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("c = 1.0, gamma = default:",metrics.accuracy_score(y_test, y_pred))

#c = 3.0
clf = svm.SVC(kernel = 'poly', C = 3.0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("c = 3.0, gamma = default:",metrics.accuracy_score(y_test, y_pred))

#
clf = svm.SVC(kernel = 'poly', C = 3.0, gamma = (1/150))
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("c = 3.0, gamma = 1 / 150:",metrics.accuracy_score(y_test, y_pred))
