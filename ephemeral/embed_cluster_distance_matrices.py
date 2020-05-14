#!/usr/bin/env python3

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.manifold import MDS


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', help='Input matrix')

    args = parser.parse_args()

    D = np.loadtxt(args.INPUT)
    groups = pd.read_csv('../data/participant_groups.csv')['cluster']

    # TODO: make embedding configurable?
    clf = MDS(random_state=5, dissimilarity='precomputed', metric=True)
    X = clf.fit_transform(D)

    X = np.append(X, groups.to_numpy().reshape(-1, 1), axis=1)

    df = pd.DataFrame(X, columns=['x', 'y', 'group'])
    df['group'] = df['group'].astype('int')

    basename = os.path.basename(args.INPUT)
    basename = os.path.splitext(basename)[0]

    df.to_csv(
        f'Embedding_{basename}.csv',
        index=False,
        float_format='%.04f',
    )

    plt.scatter(X[:, 0], X[:, 1], c=groups, cmap='Set1')
    plt.colorbar()
    plt.show()
