#!/usr/bin/env python3
#
# Embeds baseline (autocorrelation) using MDS.

import argparse
import sys

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', type=str, help='Input files')

    parser.add_argument(
        '-c', '--children',
        action='store_true',
        help='If set, only visualises children'
    )

    args = parser.parse_args()

    df = []

    for filename in sorted(args.INPUT):
        M = np.load(filename)['X']
        df.append(M.ravel())

    df = pd.DataFrame(df)
    df = (df - df.min()) / (df.max() - df.min())

    df_cohorts = pd.read_csv('../data/participant_groups.csv')
    df['cohort'] = df_cohorts['cluster']
    df['cohort'] = df['cohort'].astype('str')

    X = df.select_dtypes(np.number).to_numpy()

    df_ages = pd.read_csv('../data/participant_ages.csv')
    df['age'] = df_ages['Age']
    df['cohort'] = df['cohort'].astype(int)

    if args.children:
        X = X[df['age'] < 18]
        df = df[df['age'] < 18]

        # Just to adjust the colour map later on
        df['cohort'] -= 1

    D = pairwise_distances(X, metric='l2')
    Y = MDS(
        dissimilarity='precomputed',
        max_iter=1000,
        n_init=32,
        random_state=42,
    ).fit_transform(D)

    plt.scatter(
        x=Y[:, 0],
        y=Y[:, 1],
        c=df['cohort'].values,
        cmap='Set1',
    )

    df['x'] = Y[:, 0]
    df['y'] = Y[:, 1]

    df = df[['x', 'y', 'age', 'cohort']]
    df.to_csv(sys.stdout, index=False, float_format='%.2f')

    plt.colorbar()
    plt.show()
