import sklearn
import numpy as np
import pandas as pd
import os
from pathlib import Path

train_directory = 'C:\\Users\\brett\\source\\repos\\ML Project 1\\ML Project 1\\project1_datasets\\enron4_train\\enron4\\train'

df = pd.DataFrame(data=None, index=None, columns = ['hamspam', 'text', 'bow', 'ber'], dtype=None)

for filename in os.listdir(train_directory):
    f = os.path.join(train_directory, filename)
    if os.path.isfile(f):
        myf = open('C:\\Users\\brett\\source\\repos\\ML Project 1\\ML Project 1\\project1_datasets\\enron4_train\\enron4\\train\\' + filename, 'r', errors='replace').read()
        if (filename.find('ham') > 0):
            df.loc[len(df.index)] = [1, myf, [], []]
        else:
            df.loc[len(df.index)] = [0, myf, [], []]

#print(df.head())

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
    df.bow[i] = wiw
    df.ber[i] = wir

#
#   multinomial Naive Bayes 
#

#prior
prior = [0, 0]
text = ['', '']

for i in range(dfl):
    if (df.loc[i][0] == 1):
        prior[0] = prior[0] + 1
        text[0] = text[0] + " " +df.loc[i][1].lower()
    else:
        prior[1] = prior[1] + 1
        text[1] = text[1] +" " +df.loc[i][1].lower()
prior[0] = float(prior[0]) / float(dfl)
prior[1] = float(prior[1]) / float(dfl)

#print(prior[0])
#print(prior[1])

#condprob
condprob = [np.zeros(len(w)), np.zeros(len(w))]
tct = [np.zeros(len(w)), np.zeros(len(w))]
for i in range(2):
    for ti in text[i].split():
        if (ti.isnumeric()):
            ti = "xnumeric_valuex"
        ei = w.index(ti)
        tct[i][ei] = tct[i][ei] + 1

sumtct = [sum(tct[0]) + 1, sum(tct[1]) + 1]
#print(sumtct)

for i in range(2):
    for j in range(len(w)):
        condprob[i][j] = (tct[i][j] + 1) / sumtct[i]

test_directory = 'C:\\Users\\brett\\source\\repos\\ML Project 1\\ML Project 1\\project1_datasets\\enron4_test\\enron4\\test'

test_df = pd.DataFrame(data=None, index=None, columns = ['hamspam', 'text', 'bow', 'ber'], dtype=None)

for filename in os.listdir(test_directory):
    f = os.path.join(test_directory, filename)
    if os.path.isfile(f):
        myf = open('C:\\Users\\brett\\source\\repos\\ML Project 1\\ML Project 1\\project1_datasets\\enron4_test\\enron4\\test\\' + filename, 'r', errors='replace').read()
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
    #false positive warnings from python here
    test_df.bow[i] = wiw
    test_df.ber[i] = wir
print("test2")
score = [np.zeros(test_dfl), np.zeros(test_dfl)]

for i in range(test_dfl):
    score[0][i] = np.log(prior[0])
    score[1][i] = np.log(prior[1])
    for t in test_df.text[i].split():
        t = t.lower()
        if (t.isnumeric()):
            t = "xnumeric_valuex"
        try:
            j = w.index(t)
            score[0][i] = score[0][i] + np.log(condprob[0][j])
            score[1][i] = score[1][i] + np.log(condprob[1][j])
        except:
            pass

classified = np.zeros(test_dfl)

for i in range(test_dfl):
    if (score[0][i] > score[1][i]):
        classified[i] = 1
    else:
        classified[i] = 0

truth = test_df.hamspam
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
