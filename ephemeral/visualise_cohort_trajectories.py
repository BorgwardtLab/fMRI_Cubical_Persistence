#!/usr/bin/env python3
#
# Visualises persistence-based embeddings in the form of trajectories
# in a space of a given dimension. This is a simplified version of the
# generic embedding script, focussing on the relevant scenarios for a
# publication. Moreover, only a single trajectory is calculated per
# cohort.

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from phate import PHATE
from m_phate import M_PHATE

from tqdm import tqdm


def foo():
    X = np.array([row for row in data])

    if rolling is not None:
        df = pd.DataFrame(X)
        df = df.rolling(rolling, axis=0, min_periods=1).mean()

        X = df.to_numpy()

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    # TODO: decide whether this will be useful or not
    # X -= np.mean(X, axis=0)

    if metric is not None:
        X = pairwise_distances(X, metric=metric)

        # TODO: check whether the classifier supports this
        encoder.set_params(dissimilarity='precomputed')

    try:
        if refit:
            X = encoder.fit_transform(X)
        else:
            X = encoder.transform(X)
    except ValueError:
        # Nothing to do but to move on...
        return

    colours = np.linspace(0, 1, len(X))

    if args.dimension == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots()

    fig.set_size_inches(5, 5)
    ax.set_title(subject)

    min_x = X[:, 0].min()
    max_x = X[:, 0].max()
    min_y = X[:, 1].min()
    max_y = X[:, 1].max()

    if args.dimension == 3:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colours)

        min_z = X[:, 2].min()
        max_z = X[:, 2].max()

        ax.set_zlim(min_z, max_z)

    elif args.dimension == 2:
        ax.scatter(X[:, 0], X[:, 1], c=colours, cmap='Spectral')

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Create output directories for storing *all* subjects in. This
    # depends on the input file.
    os.makedirs(path, exist_ok=True)
    plt.tight_layout()

    name = ''

    if prefix is not None:
        name += f'{prefix}'

    name += f'_{args.dimension}D'

    if rolling is not None:
        name += f'_r{rolling}'
    else:
        name += f'_r0'

    # TODO: this cannot handle callable arguments yet, but at least some
    # simple defaults.
    if type(metric) is str:
        name += f'_{metric}'

    name += f'_{subject}'

    if args.global_embedding:
        name += '_global'

    plt.savefig(
        os.path.join(path, f'{name}.png'),
        bbox_inches='tight'
    )

    # Save the raw data as well
    df = pd.DataFrame(X)
    df.index.name = 'time'

    if args.dimension == 3:
        df.columns = ['x', 'y', 'z']
    elif args.dimension == 2:
        df.columns = ['x', 'y']

    df.to_csv(
        os.path.join(path, f'{name}.csv'),
        float_format='%.04f',
        index=True
    )

    # FIXME: make density calculation configurable
    if args.dimension == 2 and False:
        ax.clear()
        ax.set_title(subject)

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        ax.hist2d(X[:, 0], X[:, 1], bins=20, cmap='viridis')

        plt.tight_layout()
        plt.savefig(
            os.path.join(path, f'{name}_density.png'),
            bbox_inches='tight'
        )

    plt.close(fig)


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
        '-m', '--metric',
        help='Specifies metric for calculating embedding',
        type=str,
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
