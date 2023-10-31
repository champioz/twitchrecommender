import sys
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from pandas.api.types import CategoricalDtype

np.random.seed(13)


def clean_users(matrix):
    matrix['Duration'] = (matrix['End'] - matrix['Start']) * 10
    matrix = matrix[['User', 'Channel', 'Duration']
                    ].groupby(['User', 'Channel'], as_index=False).agg('sum')
    totals = matrix[['User', 'Duration']].groupby(
        'User').agg('sum').reset_index()
    matrix['Proportion'] = matrix.Duration.div(
        matrix.User.map(totals.set_index('User')['Duration']))
    matrix.drop('Duration', axis=1, inplace=True)
    return matrix


def discretize(matrix):
    matrix[matrix != 0] = 1
    return matrix


# Load two full datasets
user = pd.read_csv('./data_test/100k_a.csv',
                   names=['User', 'Stream', 'Channel', 'Start', 'End'])
chan = pd.read_csv('./data_test/master.csv')
print(chan['Channel'].tolist())
chan['Channel'] = chan['Channel'].str.lower()


# Get channels present in each dataset
user_channels = user['Channel'].str.lower().tolist()
chan_channels = chan['Channel'].str.lower().tolist()

print(user.shape)
# Find intersection
intersection_channels = list(
    set(user_channels).intersection(set(chan_channels)))
user = user[user['Channel'].isin(intersection_channels)]
print(user.shape)


# Split user data to test and train, sorted temporally
user.sort_values(by='Start', inplace=True)
m = int(np.floor(user.shape[0] * 0.7))
user_train = user.iloc[0:m, :]
user_test = user.iloc[m:, :]


# Clean up user datasets
user_train = clean_users(user_train.copy())
user_test = clean_users(user_test.copy())


# Populate full-width user matrix
users_tr = user_train['User'].unique()
chan_tr = user_train['Channel'].unique()
with open('./data_test/user_train_index.txt', 'w') as f:
    for line in users_tr:
        f.write(f'{line}\n')
with open('./data_test/user_train_cols.txt', 'w') as f:
    for line in chan_tr:
        f.write(f'{line}\n')

vals_tr = user_train['Proportion'].tolist()
row = user_train.User.astype(CategoricalDtype(categories=users_tr)).cat.codes
col = user_train.Channel.astype(CategoricalDtype(categories=chan_tr)).cat.codes
user_train = csr_matrix(
    (vals_tr, (row, col)), shape=(len(users_tr), len(chan_tr)))

users_te = user_test['User'].unique()
chan_te = user_test['Channel'].unique()
with open('./data_test/user_test_index.txt', 'w') as f:
    for line in users_te:
        f.write(f'{line}\n')
with open('./data_test/user_test_cols.txt', 'w') as f:
    for line in chan_te:
        f.write(f'{line}\n')

vals_te = user_test['Proportion'].tolist()
row = user_test.User.astype(CategoricalDtype(categories=users_te)).cat.codes
col = user_test.Channel.astype(CategoricalDtype(categories=chan_te)).cat.codes
user_test = csr_matrix(
    (vals_te, (row, col)), shape=(len(users_te), len(chan_te)))

# Delete 0 rows
user_train = user_train[user_train.sum(axis=1).A1 > 0, :]
user_test = user_test[user_test.sum(axis=1).A1 > 0, :]
user_train = user_train[:, user_train.sum(axis=0).A1 > 0]
user_test = user_test[:, user_test.sum(axis=0).A1 > 0]

save_npz('./data_test/user_train.npz', user_train)
save_npz('./data_test/user_test.npz', user_test)

user_train = discretize(user_train)
user_test = discretize(user_test)

save_npz('./data_test/user_train_disc.npz', user_train)
save_npz('./data_test/user_test_disc.npz', user_test)


# Remove channels not in intersection
chan.set_index('Channel', inplace=True)
removed = chan.loc[intersection_channels, :]
removed.iloc[:, 0:21].to_csv('./data_test/chan_excluded.csv')

# Separate games matrix from stats
chan.drop(chan.index.difference(intersection_channels),
          axis=0, inplace=True)

stats = chan.iloc[:, 0:21]
stats.to_csv('./data_test/stats_reduced.csv')
games = chan.iloc[:, 21:]
save_npz('./data_test/games_reduced.npz', csr_matrix(games))
with open('./data_test/channel_cols.txt', 'w') as f:
    for line in games.columns:
        f.write(f'{line}\n')
with open('./data_test/channel_index.txt', 'w') as f:
    for line in games.index:
        f.write(f'{line}\n')

stats.drop(stats.columns.difference(
    ['Language']), axis=1, inplace=True)

games_disc = discretize(games)

onehot = pd.get_dummies(stats['Language'])
stats.drop('Language', axis=1, inplace=True)
stats = stats.join(onehot)
games_disc = games_disc.merge(stats, left_index=True, right_index=True)

save_npz('./data_test/games_reduced_disc.npz', csr_matrix(games_disc))
