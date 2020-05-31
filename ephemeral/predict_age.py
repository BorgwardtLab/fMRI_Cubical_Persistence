#!/usr/bin/env python3

import argparse
import json

import pandas as pd
import numpy as np

from scipy.stats import pearsonr

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from tqdm import tqdm


def pearson_correlation(y_true, y_pred):
    """Evaluate Pearson's correlation between samples."""
    # This is ignores the $p$-value, but said value is not reliable
    # anyway given our small sample sizes.
    return pearsonr(y_true, y_pred)[0]


def summary_to_feature_matrix(filename, summary):
    """Convert summary statistics to feature matrix."""
    with open(filename) as f:
        data = json.load(f)

    X = []

    for subject in sorted(data.keys()):
        # Skip everything that is not a subject
        try:
            _ = int(subject)
        except ValueError:
            continue

        curve = data[subject][summary]
        X.append(curve)

    return np.asarray(X)


def descriptor_to_feature_matrix(filename):
    """Convert topological feature descriptor to feature matrix."""
    with open(filename) as f:
        data = json.load(f)

    X = []

    for subject in sorted(data.keys()):
        # Skip everything that is not a subject
        try:
            _ = int(subject)
        except ValueError:
            continue

        # Unravel the descriptor and consider it to be a single row in
        # the matrix.
        curve = np.asarray(data[subject]).ravel()
        X.append(curve)

    return np.asarray(X)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, nargs='+')
    parser.add_argument('-s', '--summary', type=str)
    parser.add_argument(
        '-c', '--children',
        action='store_true',
        help='If set, restricts prediction to children'
    )

    args = parser.parse_args()

    if len(args.INPUT) != 1:
        X = np.vstack([np.load(f)['X'].ravel() for f in sorted(args.INPUT)])
    else:
        if args.INPUT[0].endswith('.npy'):
            X = np.load(args.INPUT[0])

            # TODO: not sure whether this is the smartest way of
            # reshaping the data.
            X = X.reshape(X.shape[0], -1)
        elif args.summary is not None:
            X = summary_to_feature_matrix(args.INPUT[0], args.summary)
        else:
            X = descriptor_to_feature_matrix(args.INPUT[0])

    # Arbitrary threshold, need that so that we do not have to wait too
    # long for the results.
    if X.shape[1] > 1000:
        X = PCA(n_components=10).fit_transform(X)

    y = pd.read_csv('../data/participant_ages.csv')['Age'].values

    if args.children:
        child_indices = np.nonzero(y < 18)
        y = y[child_indices]
        X = X[child_indices]

    pipeline = Pipeline(
        steps=[
            ('scaler', StandardScaler()),
            ('clf', RidgeCV())
        ]
    )

    loo = LeaveOneOut()
    y_pred = []

    for train_index, test_index in tqdm(loo.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        pipeline.fit(X_train, y_train)
        y_pred.append(*pipeline.predict(X_test))

    print(f'R^2: {r2_score(y,  y_pred):.2f}')
    print(f'Correlation coefficient: {pearson_correlation(y, y_pred):.2f}')
    print(f'MSE: {mean_squared_error(y, y_pred):.2f}')