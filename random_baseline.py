import csv
import sys
import pandas as pd
import numpy as np
import random
from scipy.sparse import load_npz
from scipy.spatial import distance
from collections import defaultdict

np.random.seed(13)


def process_labels(matrix, index):

    ind, col = matrix.nonzero()
    col_labels = range(matrix.shape[1])

    label_dict = defaultdict(list)
    for i, j in zip(ind, col):
        label_dict[col_labels[j]].append(index[i])

    return label_dict


def update_scores(candidates, scores, weights):
    updated = {}
    for ind, c in enumerate(candidates):
        updated[c] = float(scores[c]) * weights[ind]
    recs = [k for k, v in sorted(
        updated.items(), key=lambda item: item[1]
    )][::-1]
    n = min(5, len(recs))
    return list(recs[:n])


def fetch_scores(candidates, scores):
    updated = {}
    for c in candidates:
        updated[c] = float(scores[c])
    recs = [k for k, v in sorted(
        updated.items(), key=lambda item: item[1]
    )][::-1]
    n = min(5, len(recs))
    return list(recs[:n])


def collect_metrics(recs, response, train):
    print(response)

    novel = set(response).difference(set(train))
    repeat = set(response).intersection(set(train))

    ACC = len((set(recs) & set(response))) / len(response)

    results = {}
    results['ACC'] = ACC

    if len(novel) > 0:
        ACC_novel = len((set(recs) & novel)) / len(novel)
        results['ACC_novel'] = ACC_novel

    if len(repeat) > 0:
        ACC_repeat = len((set(recs) & repeat)) / len(repeat)
        results['ACC_repeat'] = ACC_repeat

    return results


def factorize(df):
    x = df.stack()
    x[:] = x.factorize()[0]
    return x.unstack()


def upper_tri_avg(df):
    df = df.where(np.triu(np.ones(df.shape)).astype(bool)).stack()
    return df.mean()


# Load test user data

train = load_npz('./data/user_train.npz')
test = load_npz('./data/user_test.npz')

with open('./data/user_test_index.txt', 'r') as f:
    user_test_index = [line.strip() for line in f]
with open('./data/user_test_cols.txt', 'r') as f:
    user_test_cols = [line.strip() for line in f]
with open('./data/channel_index.txt', 'r') as f:
    chan_index = [line.strip() for line in f]
with open('./data/user_train_index.txt', 'r') as f:
    user_train_index = [line.strip() for line in f]
with open('./data/user_train_cols.txt', 'r') as f:
    user_train_cols = [line.strip() for line in f]
with open('./outputs/scores/channel_scores.csv', 'r') as f:
    reader = csv.reader(f)
    channel_scores = {rows[1]: rows[2] for rows in reader}

ind, cols = train.nonzero()

traindict = defaultdict(dict)
traindict_disc = defaultdict(list)
for i, j in zip(ind, cols):
    traindict[user_train_index[i]][user_train_cols[j]] = train[i, j]
    traindict_disc[user_train_index[i]].append(user_train_cols[j])


ind, cols = test.nonzero()

testdict = defaultdict(list)
for i, j in zip(ind, cols):
    testdict[user_test_index[i]].append(user_test_cols[j])


DIAGNOSTICS = {}
ACCS_prob = []
ACCS_prob_novel = []
ACCS_prob_repeat = []
PROB = {}
for user in traindict.keys():

    recs_prob = np.random.choice(list(channel_scores.keys()), 5, replace=False)

    PROB[user] = recs_prob

    results_prob = collect_metrics(
        recs_prob, testdict[user], traindict[user])

    ACCS_prob.append(results_prob['ACC'])
    try:
        ACCS_prob_novel.append(results_prob['ACC_novel'])
    except KeyError:
        pass
    try:
        ACCS_prob_repeat.append(results_prob['ACC_repeat'])
    except KeyError:
        pass

PROB_fact = factorize(pd.DataFrame(PROB))

prob_personalization = 1 - upper_tri_avg(
    pd.DataFrame(1 - distance.cdist(
        PROB_fact,
        PROB_fact,
        metric='cosine'
    )))
mACC_prob = round(np.mean(ACCS_prob), 4)
mACC_prob_novel = round(np.mean(ACCS_prob_novel), 4)
mACC_prob_repeat = round(np.mean(ACCS_prob_repeat), 4)
DIAGNOSTICS['random'] = {
    'Probability ACC': mACC_prob,
    'Probability Novel ACC': mACC_prob_novel,
    'Probability Repeat ACC': mACC_prob_repeat,
    'Probability Personalization': prob_personalization
}
pd.DataFrame(DIAGNOSTICS).T.to_csv('./outputs_test/random.csv')
