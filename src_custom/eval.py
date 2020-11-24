"""
Evaluation script for a variety of datasets.
This is a slightly modified version of the original
skip-thoughts eval code, found here:

https://github.com/ryankiros/skip-thoughts/blob/master/eval_classification.py
"""
import logging
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

_LOGGER = logging.getLogger(__name__)


def load_encode_data(predict_vec_func, name, loc, seed=1, test_batch_size=100):
    """load in a binary classification dataste for evaluation"""

    if name == 'MR':
        with open('data/rt10662/rt-polarity.pos', 'rb') as f:
            pos = [line.decode('latin-1').strip() for line in f]
        with open(os.path.join(loc, 'rt10662/rt-polarity.neg'), 'rb') as f:
            neg = [line.decode('latin-1').strip() for line in f]
    elif name == 'ET_sent':
        df = pd.read_csv('data/et_valence/laused.tsv', sep='\t', header=None)
        pos = df[df[0].isin(['pos', 'neu'])][1].tolist()
        neg = df[df[0] == 'neg'][1].tolist()
    elif name == 'SUBJ':
        with open(os.path.join(loc, 'plot.tok.gt9.5000'), 'rb') as f:
            pos = [line.decode('latin-1').strip() for line in f]
        with open(os.path.join(loc, 'quote.tok.gt9.5000'), 'rb') as f:
            neg = [line.decode('latin-1').strip() for line in f]
    elif name == 'CR':
        with open(os.path.join(loc, 'customerr/custrev.pos'), 'rb') as f:
            pos = [line.decode('latin-1').strip() for line in f]
        with open(os.path.join(loc, 'customerr/custrev.neg'), 'rb') as f:
            neg = [line.decode('latin-1').strip() for line in f]
    elif name == 'MPQA':
        with open(os.path.join(loc, 'mpqa/mpqa.pos'), 'rb') as f:
            pos = [line.decode('latin-1').strip() for line in f]
        with open(os.path.join(loc, 'mpqa/mpqa.neg'), 'rb') as f:
            neg = [line.decode('latin-1').strip() for line in f]

    labels = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
    text, labels = shuffle(pos + neg, labels, random_state=seed)
    size = len(labels)
    _LOGGER.info("Loaded dataset {} with total lines: {}".format(name, size))

    features = predict_vec_func(text)
    # _LOGGER.info("Processing {:5d} batches of size {:5d}".format(len(feature_list), test_batch_size))
    # features = np.concatenate(feature_list)
    _LOGGER.info("Test feature matrix of shape: {}".format(features.shape))

    return text, labels, features


def fit_clf(X_train, y_train, X_test, y_test, s):
    """Fits a single classifier and returns test accuracy"""
    clf = LogisticRegression(solver='sag', C=s)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    # _LOGGER.info("Fitting logistic model with s: {:3d} and acc: {:.2%}".format(s, acc))
    return acc


def test_performances(predict_vec_func, datasets=['MR'], loc='', seed=1):
    accs = []
    for dataset in datasets:
        text, labels, features = load_encode_data(predict_vec_func, dataset, loc=loc, seed=seed)
        X_train, X_test, y_train, y_test = train_test_split(features, labels)
        acc_t = fit_clf(X_train, y_train, X_test, y_test, 1)
        accs.append((acc_t, dataset))
        _LOGGER.info("Trained on {:4d} examples - Test Accuracy: {:.2%}".format(len(X_train), acc_t))
    return accs
