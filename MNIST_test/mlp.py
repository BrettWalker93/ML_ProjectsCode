import numpy as np

X_train = np.genfromtxt('xtrain.csv', delimiter=',',dtype=float)
X_test = np.genfromtxt('xtest.csv', delimiter=',',dtype=float)
y_train = np.genfromtxt('ytrain.csv', delimiter=',',dtype=float)
y_test = np.genfromtxt('ytest.csv', delimiter=',',dtype=float)

from sklearn.neural_network import MLPClassifier
from sklearn import metrics

#Default MLPClassifier
print("MLPClassifier:")
print("Activation = ReLU:")
clf = MLPClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("default parameters:",metrics.accuracy_score(y_test, y_pred))

clf = MLPClassifier(alpha=0.005, learning_rate = 'adaptive')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("alpha=0.005, adaptive learning rate:",metrics.accuracy_score(y_test, y_pred))

clf = MLPClassifier(learning_rate = 'adaptive')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("adaptive learning rate:",metrics.accuracy_score(y_test, y_pred))

clf = MLPClassifier(alpha = 0.005)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("alpha = 0.005:",metrics.accuracy_score(y_test, y_pred))

print("Activation = identity:")
clf = MLPClassifier(activation='identity')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("default parameters:",metrics.accuracy_score(y_test, y_pred))

clf = MLPClassifier(activation='identity', learning_rate = 'adaptive')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("adaptive learning rate:",metrics.accuracy_score(y_test, y_pred))

clf = MLPClassifier(activation='identity', alpha=0.005)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("alpha = 0.005:",metrics.accuracy_score(y_test, y_pred))

clf = MLPClassifier(activation='identity', alpha=0.005, learning_rate = 'adaptive')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("alpha = 0.005, adaptive learning rate:",metrics.accuracy_score(y_test, y_pred))

print("Activation = identity:")
clf = MLPClassifier(activation='logistic')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("default parameters:",metrics.accuracy_score(y_test, y_pred))

clf = MLPClassifier(activation='logistic', learning_rate = 'adaptive')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("adaptive learning rate:",metrics.accuracy_score(y_test, y_pred))

clf = MLPClassifier(activation='logistic', alpha=0.005)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("alpha = 0.005:",metrics.accuracy_score(y_test, y_pred))

clf = MLPClassifier(activation='logistic', alpha=0.005, learning_rate = 'adaptive')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("alpha = 0.005, adaptive learning rate:",metrics.accuracy_score(y_test, y_pred))

clf = MLPClassifier(activation='tanh')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("default parameters:",metrics.accuracy_score(y_test, y_pred))

clf = MLPClassifier(activation='tanh', learning_rate = 'adaptive')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("adaptive learning rate:",metrics.accuracy_score(y_test, y_pred))

clf = MLPClassifier(activation='tanh', alpha=0.005)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("alpha = 0.005:",metrics.accuracy_score(y_test, y_pred))

clf = MLPClassifier(activation='tanh', alpha=0.005, learning_rate = 'adaptive')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("alpha = 0.005, adaptive learning rate:",metrics.accuracy_score(y_test, y_pred))

print("Activation = ReLU:")
clf = MLPClassifier(hidden_layer_sizes=(150,100,50))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("alpha=0.005, adaptive learning rate:",metrics.accuracy_score(y_test, y_pred))

print("Activation = identity:")
clf = MLPClassifier(alpha=0.005, learning_rate = 'adaptive', hidden_layer_sizes=(150,100,50), activation='identity')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("alpha=0.005, adaptive learning rate:",metrics.accuracy_score(y_test, y_pred))

print("Activation = logistic:")
clf = MLPClassifier(alpha=0.005, learning_rate = 'adaptive', hidden_layer_sizes=(150,100,50), activation='logistic')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("alpha=0.005, adaptive learning rate:",metrics.accuracy_score(y_test, y_pred))

print("Activation = tanh:")
clf = MLPClassifier(alpha=0.005, hidden_layer_sizes=(150,100,50), activation='tanh')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("alpha=0.005:",metrics.accuracy_score(y_test, y_pred))