import pandas as pd
import numpy as np
import sys
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(13)
np.set_printoptions(formatter={'float_kind': '{:f}'.format})


def preprocess(df):

    df.columns = map(str.lower, df.columns)
    replacements = {' ': '_', '(': '', ')': '', ':': ''}
    for old, new in replacements.items():
        df.columns = df.columns.str.replace(old, new, regex=False)

    days = ['monday', 'tuesday', 'wednesday', 'thursday',
            'friday', 'saturday', 'sunday']
    df.set_index('channel', inplace=True)
    df.drop(['unnamed_0', 'language', 'watch_time_mins',
             'average_followers_per_stream'],
            axis=1, inplace=True)
    #df.drop(days, axis=1, inplace=True)
    return df


train = pd.read_csv('./data_test/chan_excluded.csv').dropna()
test = pd.read_csv('./data_test/stats_reduced.csv').dropna()
train = preprocess(train.copy())
test = preprocess(test.copy())

trainX, trainY = train.loc[:, train.columns !=
                           'followers_gained'], train['followers_gained']
testX, testY = test.loc[:, test.columns !=
                        'followers_gained'], test['followers_gained']

mod = sm.OLS(trainY, trainX.astype(float)).fit()


DIAGNOSTICS = {}

RF = RandomForestRegressor(
    n_estimators=500, oob_score=True).fit(trainX, trainY)
res = cross_validate(RF, trainX, trainY,
                     cv=15,
                     return_train_score=True,
                     scoring=['neg_mean_squared_error',
                              'r2'])
RF_preds = RF.predict(testX)
RF_mse = mean_squared_error(RF_preds, testY)
RF_r2 = r2_score(RF_preds, testY)
DIAGNOSTICS['RandomForest'] = {
    'Train R2': round(np.mean(res['train_r2']), 3),
    'Train MSE': round(np.mean(res['train_neg_mean_squared_error']), 3),
    'CV R2': round(np.mean(res['test_r2']), 3),
    'CV MSE': round(np.mean(res['test_neg_mean_squared_error']), 3),
    'Test MSE': round(RF_mse),
    'Test R2:': round(RF_r2)
}
pd.DataFrame({'Pred': RF_preds, 'Resp': testY}).to_csv(
    './outputs/RF_rank_scores.csv')

NN = MLPRegressor().fit(trainX, trainY)
res = cross_validate(NN, trainX, trainY,
                     cv=15,
                     return_train_score=True,
                     scoring=['neg_mean_squared_error',
                              'r2'])
NN_preds = NN.predict(testX)
NN_mse = mean_squared_error(NN_preds, testY)
NN_r2 = r2_score(NN_preds, testY)
DIAGNOSTICS['NeuralNet'] = {
    'Train R2': round(np.mean(res['train_r2']), 3),
    'Train MSE': round(np.mean(res['train_neg_mean_squared_error']), 3),
    'CV R2': round(np.mean(res['test_r2']), 3),
    'CV MSE': round(np.mean(res['test_neg_mean_squared_error']), 3),
    'Test MSE': round(NN_mse),
    'Test R2:': round(NN_r2)
}

GB = GradientBoostingRegressor().fit(trainX, trainY)
res = cross_validate(GB, trainX, trainY,
                     cv=15,
                     return_train_score=True,
                     scoring=['neg_mean_squared_error',
                              'r2'])
GB_preds = GB.predict(testX)
GB_mse = mean_squared_error(GB_preds, testY)
GB_r2 = r2_score(GB_preds, testY)
DIAGNOSTICS['GradientBoost'] = {
    'Train R2': round(np.mean(res['train_r2']), 3),
    'Train MSE': round(np.mean(res['train_neg_mean_squared_error']), 3),
    'CV R2': round(np.mean(res['test_r2']), 3),
    'CV MSE': round(np.mean(res['test_neg_mean_squared_error']), 3),
    'Test MSE': round(GB_mse),
    'Test R2:': round(GB_r2)
}

LG = LinearRegression().fit(trainX, trainY)
res = cross_validate(LG, trainX, trainY,
                     cv=15,
                     return_train_score=True,
                     scoring=['neg_mean_squared_error',
                              'r2'])
LG_preds = LG.predict(testX)
LG_mse = mean_squared_error(LG_preds, testY)
LG_r2 = r2_score(LG_preds, testY)
DIAGNOSTICS['LinearRegression'] = {
    'Train R2': round(np.mean(res['train_r2']), 3),
    'Train MSE': round(np.mean(res['train_neg_mean_squared_error']), 3),
    'CV R2': round(np.mean(res['test_r2']), 3),
    'CV MSE': round(np.mean(res['test_neg_mean_squared_error']), 3),
    'Test MSE': round(LG_mse),
    'Test R2:': round(LG_r2)
}


output = pd.DataFrame.from_dict(DIAGNOSTICS).T
output.to_csv('./outputs/ranking_scores.csv')
