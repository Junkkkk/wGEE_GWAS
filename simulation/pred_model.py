from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def split_train_test(samplingdata, test_ratio=0.1):
    train_X, test_X, train_Y, test_Y = train_test_split(samplingdata[0], samplingdata[1], test_size=test_ratio)

    #Y : 0 / 2 -> 0 / 1
    le = preprocessing.LabelEncoder()
    le.fit(train_Y)
    train_Y=le.transform(train_Y)

    le.fit(test_Y)
    test_Y=le.transform(test_Y)

    return [train_X, test_X, train_Y, test_Y]

def train_model(split_data):
    train_X = split_data[0][:,:-1]
    train_Y = split_data[2]

    model = LogisticRegression()

    params = {
        # 'penalty': ['elasticnet'],
        # 'solver': ['saga'],
        # 'C': [0.001, 0],
        # 'l1_ratio': [0, 0.5, 1.0],
        'max_iter': [100],
        #'max_iter': [None],
        #'class_weight': ['balanced']
    }


    # model = RandomForestClassifier()
    #
    # params = {
    #     'n_estimators': [100]
    # }
    # params = {
    #     'max_depth': [10, 50, 100, 500],
    #     'min_samples_leaf': [1, 5, 10, 20],
    #     'n_estimators': [100, 300, 500]
    # }

    model_grid = GridSearchCV(model,
                              param_grid=params, cv=5, scoring='roc_auc',
                              verbose=0, n_jobs=1)
    model_grid.fit(train_X, train_Y)

    return model_grid

def evaluate_model(model, split_data):
    test_X = split_data[1][:,:-1]
    test_Y = split_data[3]

    probs = model.predict_proba(test_X)[:, 1]
    #pred = (model.predict_proba(test_X)[:, 1] >= cutoff).astype(bool)
    pred = model.predict(test_X)

    tn, fp, fn, tp = confusion_matrix(list(test_Y), pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    acc = (tp + tn) / (tn + fp + fn + tp)
    auc = roc_auc_score(list(test_Y), probs)

    return auc, acc, sen, spe

def predict_mci(model, samplingdata):
    return model.predict_proba(samplingdata[2][:,:-1])[:,1]

def evaluate_mci(model, samplingdata):
    test_X = samplingdata[2][:,:-1]
    test_Y = samplingdata[3]

    le = preprocessing.LabelEncoder()
    le.fit(test_Y)
    test_Y=le.transform(test_Y)

    probs = model.predict_proba(test_X)[:, 1]
    pred = model.predict(test_X)

    # print(test_Y)
    # print(pred)
    tn, fp, fn, tp = confusion_matrix(list(test_Y), pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    acc = (tp + tn) / (tn + fp + fn + tp)
    auc = roc_auc_score(list(test_Y), probs)

    return auc, acc, sen, spe

def find_optimal_cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])
