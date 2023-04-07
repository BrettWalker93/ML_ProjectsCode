import numpy as np
import numba as nb
import pandas as pd
from tqdm.auto import tqdm

def average_vote(votes):
    return (np.mean(votes))

train_data = np.genfromtxt('C:\\Users\\brett\\source\\repos\\MLProject2\\MLProject2\\netflix\\TrainingRatings.txt', 
                           delimiter=',',dtype=int)

train_df = pd.DataFrame(train_data, columns = ['user', 'title', 'rating'])
num_users = train_df['user'].nunique()
unique_ids = train_df['user'].unique()

train_df['user'] = train_df['user'].transform(lambda x: np.where(unique_ids==x)[0][0])

average_votes = train_df.groupby(['user'])['rating'].transform('mean')

train_df['delta'] = train_df['rating'] - average_votes
train_df['d_sq'] = np.square(train_df['delta'])

def calculate_weight(a):

    usera = train_df.loc[train_df['user'] == a]
    movies = usera['title'].to_numpy(dtype=np.int64)

    others = train_df[(train_df['title'].isin(train_df.loc[(train_df.user == a), 'title'])) & (train_df.user!=a)]

    ua = usera.to_numpy(dtype=np.float64)
    ui = others.to_numpy(dtype=np.float64)

    r = usera.shape[0]
    k = others.shape[0]

    x = calc_loop(movies, ua, ui, k)

    return x

@nb.jit(nopython=True)
def calc_loop(movies, ua, ui, k):

    weights = np.zeros(num_users, np.float64)
    denoma = np.zeros(num_users, np.float64)
    denomi = np.zeros(num_users, np.float64)

    for n in range(k):
        aindex = -1
        for m in range(movies.size):
            if ui[n][1] == movies[m]:
                aindex = m
                break
        d = ua[aindex][3]
        print(d)
        s = ua[aindex][4]
        print(s)
        index = np.int64(ui[n][0])            
        weights[index] = weights[index] + ui[n][3]*d
        denomi[index] = denomi[index] + ui[n][4]
        denoma[index] = denoma[index] + s

    try:
        weights = weights / np.sqrt(denoma * denomi)
    except:
        pass

    return weights


correlation_matrix = np.zeros((num_users, num_users), dtype=float)

def weight_loop_nb():
    for a in tqdm(range(num_users), desc = 'constructing correlation matrix', total = num_users):
        correlation_matrix[a,] = calculate_weight(a)

weight_loop_nb()

np.savetxt("correlation_matrix2.csv", correlation_matrix, delimiter = ',')

