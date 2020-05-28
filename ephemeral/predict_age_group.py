#!/usr/bin/env python3

import argparse
import json

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tqdm import tqdm


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

    args = parser.parse_args()

    if len(args.INPUT) != 1:
        X = np.vstack([np.load(f)['X'].ravel() for f in sorted(args.INPUT)])
    else:
        if args.summary is not None:
            X = summary_to_feature_matrix(args.INPUT[0], args.summary)
        else:
            X = descriptor_to_feature_matrix(args.INPUT[0])

    # Arbitrary threshold, need that so that we do not have to wait too
    # long for the results.
    if X.shape[1] > 1000:
        X = PCA(n_components=10).fit_transform(X)

    y = pd.read_csv('../data/participant_groups.csv')['cluster'].values

    print(y)
    print(X.shape)

    loo = LeaveOneOut()
    y_pred = []

    for train_index, test_index in tqdm(loo.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        clf = SVC()
        clf.fit(X_train, y_train)

        X_test = scaler.transform(X_test)
        y_pred.append(*clf.predict(X_test))

    C = confusion_matrix(y, y_pred)
    print(C)
    print(np.trace(C), f'({100 * np.trace(C) / np.sum(C):.2f}%)')
