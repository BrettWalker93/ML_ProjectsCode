import numpy as np
import numba as nb
import pandas as pd
from tqdm.auto import tqdm

def average_vote(votes):
    return (np.mean(votes))

weights = np.genfromtxt('C:\\Users\\brett\\source\\repos\\MLProject2\\MLProject2\\correlation_matrix.csv', 
                           delimiter=',',dtype=float)

np.nan_to_num(weights, copy=False, nan=0.0)

@nb.jit
def make_kappa(w, s):
    w2 = np.zeros(s, dtype=np.float64) 

    for i in range(s):
        for j in range(s):
            w2[i] = w2[i] + np.abs(w[j][i])
        #print(w2[i])
        w2[i] = 1 / w2[i]
    return w2

kappa = make_kappa(weights, weights[0].size)

#organize data
train_data = np.genfromtxt('C:\\Users\\brett\\source\\repos\\MLProject2\\MLProject2\\netflix\\TrainingRatings.txt', 
                           delimiter=',',dtype=int)

train_df = pd.DataFrame(train_data, columns = ['user2', 'title', 'rating'])
unique_ids = train_df['user2'].unique()

average_votes = train_df.groupby(['user2'])['rating'].transform('mean')

train_df['user'] = train_df['user2'].transform(lambda x: np.where(unique_ids==x)[0][0])
train_df['delta'] = train_df['rating'] - average_votes

#fix average votes
average_votes = train_df.groupby(['user'])['rating'].mean()

train_df_trunc = train_df.drop_duplicates(subset=['user'])

test_data = np.genfromtxt('C:\\Users\\brett\\source\\repos\\MLProject2\\MLProject2\\netflix\\TestingRatings.txt', 
                           delimiter=',',dtype=int)

test_df = pd.DataFrame(test_data, columns = ['user', 'title', 'rating'])

test_df['user'] = test_df['user'].map(train_df_trunc.set_index('user2')['user'])

print(test_df.iloc[1])

print(    test_df.iloc[1]['user'])

print(  test_df.iloc[1]['title'])

input()

#predict vote for given user
def predict(a):

    #FIX: ONLY LOOK AT MOVIE IN QUESTION
    #MAYBE others = train_df[(train_df['title'].isin(train_df.loc[(train_df.title == test_df[a][1]), 'title'])) & (train_df.user!=a)]

    user = test_df.iloc[a]['user']

    title = test_df.iloc[a]['title']
    
    others = train_df.loc[train_df['title'] == title]

    #usera = train_df.loc[train_df['user'] == a]
    #movies = usera['title'].to_numpy(dtype=np.int64)
    #others = train_df[(train_df['title'].isin(train_df.loc[(train_df.user == a), 'title'])) & (train_df.user!=a)]

    ui = others.to_numpy(dtype=np.float64)

    return prediction_loop(kappa[user], average_votes[user], weights[user,], ui, ui.shape[0])

#numba loop
#@nb.jit#(nopython=True)
def prediction_loop(k, avg, w, ui, r):

    print(r)
    p = 0
    for i in range(r):
        index = np.int64(ui[i][0])
        if (index < w.size):
            p = p + w[index] * ui[i][3]

    return k * p + avg

num_predictions = test_df.shape[0]
predictions = np.zeros(num_predictions, dtype=np.float64)
def get_predictions():
    for a in tqdm(range(num_predictions), desc = 'predicting', total = num_predictions):
        predictions[a] = predict(a)

get_predictions()

np.savetxt("predictions.csv", predictions, delimiter = ',')