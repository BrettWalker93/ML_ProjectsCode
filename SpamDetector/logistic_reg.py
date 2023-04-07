
import sklearn
import numpy as np
from scipy.special import expit
import pandas as pd
import os
from pathlib import Path

np.set_printoptions(threshold=np.inf)

train_directory = 'C:\\Users\\brett\\source\\repos\\ML Project 1\\ML Project 1\\project1_datasets\\enron2_train\\enron2\\train'

df = pd.DataFrame(data=None, index=None, columns = ['hamspam', 'text', 'bow', 'ber'], dtype=None)

for filename in os.listdir(train_directory):
    f = os.path.join(train_directory, filename)
    if os.path.isfile(f):
        myf = open('C:\\Users\\brett\\source\\repos\\ML Project 1\\ML Project 1\\project1_datasets\\enron2_train\\enron2\\train\\' + filename, 'r', errors='replace').read()
        if (filename.find('ham') > 0):
            df.loc[len(df.index)] = [1, myf, [], []]
        else:
            df.loc[len(df.index)] = [0, myf, [], []]
dfl = len(df.index)
w = []
ws = []
wh = []
for i in range(dfl):
    l = df.text[i].split()
    c = df.hamspam[i]
    for e in l:
        if (e.isnumeric()):
            e = "xnumeric_valuex"
        e = e.lower()
        if not(e in w):
            w.append(e)
        if not(e in wh) and c == 1: 
            wh.append(e)
        if not(e in ws) and c == 0:
            ws.append(e)
            
for i in range(dfl):
    wiw = np.zeros(len(w))
    wir = np.zeros(len(w))
    l = df.text[i].split()
    for e in l:
        if (e.isnumeric()):
            e = "xnumeric_valuex"
        e = e.lower()
        x = w.index(e)
        wiw[x] = wiw[x] + 1
        wir[x] = 1
    #false positive warnings from python here
    #wiw[w.index("xnumeric_valuex")] = min(wiw[w.index("xnumeric_valuex")], 3)
    #wir[w.index("xnumeric_valuex")] = min(wiw[w.index("xnumeric_valuex")], 1)
    wiw = np.append(wiw, 1)
    wir = np.append(wir, 1)
    df.bow[i] = wiw
    df.ber[i] = wir

#
#   MCAP Logistic Regression algorithm with L2 regularization
#

rate = 0.01
penalty = 12
maxiter = 25

split = int(dfl * .3)

#theta = np.zeros(len(train[split]))

#train
X = []
Y = []

#valid
XX = []
YY = []

thresh = sum(df.hamspam) / dfl

for i in range(dfl):

    if (np.random.rand(1) < thresh):
        X.append(df.bow[i])
        Y.append(df.hamspam[i])
    else:
        XX.append(df.bow[i])
        YY.append(df.hamspam[i])

theta = np.random.rand(len(X[0]))
theta = (theta - 0.5) / 10000
#print(theta)
#print(Y)

def sigmoid(x):

    return 1 / (1+expit(-x))

#fit
for i in range(maxiter):

    #error
    err = Y - sigmoid(np.dot(X, theta))

    #change weights
    d_grad = rate * (np.dot(err, X) + penalty * theta)

    theta = theta + d_grad   

#
#   validation
#

#print(theta)

print("validation results")

probs = np.zeros(len(XX))

for i in range(len(XX)):
    #print(sigmoid(np.dot(XX[i],theta[0:len(theta-1)]) + theta[len(theta)-1]))
    probs[i] = sigmoid(np.dot(XX[i],theta[0:len(theta-1)]) + theta[len(theta)-1])

#probs = sigmoid(valid @ theta[0:len(theta)-1] + theta[len(theta)-1])

classified = np.round(probs)

#print(classified)
truth = YY
#print(truth)


true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0

#
test_dfl = len(XX)
#

for i in range(test_dfl):
    if (truth[i] == classified[i]):
        if (truth[i] == 1):
            true_neg = true_neg + 1
        else:
            true_pos = true_pos + 1
    else:
        if (truth[i] == 1):
            false_pos = false_pos + 1
        else:
            false_neg = false_neg + 1

#print(true_pos)
#print(true_neg)
#print(false_pos)
#print(false_neg)

#acc = (true_pos + true_neg) / test_dfl
#print("Accuracy: " + str(acc))

#prec = (true_pos / (true_pos + false_pos))
#print("Precision: " + str(prec))

#rec = (true_pos / (true_pos + false_neg))
#print("Recall: "+ str(rec))

#f1 =  (2 * prec * rec) / (prec + rec)
#print("F1 Score: " + str(f1))

#
#   testing
#
print("test results:")

test_directory = 'C:\\Users\\brett\\source\\repos\\ML Project 1\\ML Project 1\\project1_datasets\\enron2_test\\enron2\\test'

test_df = pd.DataFrame(data=None, index=None, columns = ['hamspam', 'text', 'bow', 'ber'], dtype=None)

for filename in os.listdir(test_directory):
    f = os.path.join(test_directory, filename)
    if os.path.isfile(f):
        myf = open('C:\\Users\\brett\\source\\repos\\ML Project 1\\ML Project 1\\project1_datasets\\enron2_test\\enron2\\test\\' + filename, 'r', errors='replace').read()
        if (filename.find('ham') > 0):
            test_df.loc[len(test_df.index)] = [1, myf, [], []]
        else:
            test_df.loc[len(test_df.index)] = [0, myf, [], []]

test_dfl = len(test_df.index)

for i in range(test_dfl):
    wiw = np.zeros(len(w))
    wir = np.zeros(len(w))
    l = test_df.text[i].split()
    for e in l:
        if (e.isnumeric()):
            e = "xnumeric_valuex"
        e = e.lower()
        try:
            x = w.index(e)
            wiw[x] = wiw[x] + 1
            wir[x] = 1
        except:
            pass
    wiw = np.append(wiw, 1)
    wir = np.append(wir, 1)
    #false positive warnings from python here
    test_df.bow[i] = wiw
    test_df.ber[i] = wir

#train
X = []
Y = []

for i in range(test_dfl):
    X.append(test_df.bow[i])
    Y.append(test_df.hamspam[i])
    
probs = np.zeros(len(X))

for i in range(len(X)):
    #print(sigmoid(np.dot(XX[i],theta[0:len(theta-1)]) + theta[len(theta)-1]))
    probs[i] = sigmoid(np.dot(X[i],theta))

#probs = sigmoid(valid @ theta[0:len(theta)-1] + theta[len(theta)-1])

classified = np.round(probs)

#print(classified)
truth = Y
#print(truth)


true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0

for i in range(test_dfl):
    if (truth[i] == classified[i]):
        if (truth[i] == 1):
            true_neg = true_neg + 1
        else:
            true_pos = true_pos + 1
    else:
        if (truth[i] == 1):
            false_pos = false_pos + 1
        else:
            false_neg = false_neg + 1

acc = (true_pos + true_neg) / test_dfl
print("Accuracy: " + str(acc))

prec = (true_pos / (true_pos + false_pos))
print("Precision: " + str(prec))

rec = (true_pos / (true_pos + false_neg))
print("Recall: "+ str(rec))

f1 =  (2 * prec * rec) / (prec + rec)
print("F1 Score: " + str(f1))
