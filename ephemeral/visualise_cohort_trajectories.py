#!/usr/bin/env python3
#
# Visualises persistence-based embeddings in the form of trajectories
# in a space of a given dimension. This is a simplified version of the
# generic embedding script, focussing on the relevant scenarios for a
# publication. Moreover, only a single trajectory is calculated per
# cohort.

import argparse
import json

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from phate import PHATE


def embed(Z, rolling=None, joint_embedding=False):
    """Embedding function."""
    encoder = PHATE(
        n_components=2,
        mds_solver='smacof',
        random_state=42,
        n_pca=80,
    )

    # Will be filled with the lower-dimensional representations of each
    # cohort. This makes visualising everything easier.
    df = []

    if joint_embedding:
        X = encoder.fit_transform(np.vstack(Z))
        df = pd.DataFrame(X, columns=['x', 'y'])
    else:
        for cohort in Z:

            if rolling is not None:
                cohort = pd.DataFrame(cohort).rolling(
                            rolling,
                            axis=0,
                            min_periods=1
                        ).mean()

            X = encoder.fit_transform(cohort)
            df.append(X)

        df = np.concatenate(df)
        df = pd.DataFrame(df, columns=['x', 'y'])

    n = Z.shape[0]  # number of cohorts
    m = Z.shape[1]  # number of time steps

    df['cohort'] = np.array([[i] * m for i in range(n)]).ravel()
    df['time'] = np.array(list(np.arange(m)) * n).ravel()

    g = sns.FacetGrid(
            df,
            col='cohort',
            hue='time',
            palette='Spectral',
            sharex=False,
            sharey=False)

    g = g.map(plt.scatter, 'x', 'y')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'INPUT',
        help='Input file (JSON)',
        type=str
    )

    parser.add_argument(
        '-j', '--joint-embedding',
        action='store_true',
        help='If set, calculates *joint* embeddings instead of separate ones'
    )

    parser.add_argument(
        '-r', '--rolling',
        type=int,
        default=None,
        help='If set, performs rolling average calculation'
    )

    args = parser.parse_args()

    with open(args.INPUT) as f:
        data = json.load(f)

    # Filter subjects; this could be solved smarter...
    subjects = data.keys()
    subjects = [subject for subject in subjects if len(subject) == 3]

    df_groups = pd.read_csv('../data/participant_groups.csv')

    X = np.concatenate(
        [np.array([data[subject]]) for subject in subjects]
    )

    y = df_groups['cluster'].values
    cohorts = sorted(set(y))

    assert X.shape[0] == len(y)

    # Regardless of the operating mode, we need a mean representation of
    # each cohort. This reduces the dimension of our tensor from the no.
    # of participants to the number of cohorts.
    Z = np.stack(
        [np.mean(X[y == cohort], axis=0) for cohort in cohorts],
    )

    assert Z.shape[0] == len(cohorts)

    embed(
        Z,
        args.rolling,
        args.joint_embedding
    )
