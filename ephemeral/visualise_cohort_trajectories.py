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
    """Main embedding function."""
    encoder = PHATE(
        n_components=2,
        mds_solver='smacof',
        random_state=42,
    )

    # Will be filled with the lower-dimensional representations of each
    # cohort. This makes visualising everything easier.
    df = []

    if joint_embedding:
        pass
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
            palette='Spectral')

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

    embed(Z, args.rolling, args.joint_embedding)

    if args.global_embedding:
        if args.dimension == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots()

        # Fit the estimator on *all* subjects first, then attempt to
        # embed them individually.
        #
        # M-PHATE gets special treatment because it is the only method
        # capable of handling tensors correctly.
        if args.encoder == 'm-phate':
            X = np.concatenate(
                [np.array([data[subject]]) for subject in subjects]
            )

            X = np.moveaxis(X, 1, 0)

            # Create verbose variables for this encoder; this will not
            # be used for other embeddings.
            n_time_steps = X.shape[0]
            n_samples = X.shape[1]
            n_features = X.shape[2]
        else:
            X = np.concatenate(
                [np.array(data[subject]) for subject in subjects]
            )

        X = encoder.fit_transform(X)

        # Create array full of subject indices. We can use this to
        # colour-code subjects afterwards.
        indices = np.concatenate([
            np.full(fill_value=int(subject),
                    shape=np.array(data[subject]).shape[0])
            for subject in subjects
        ])

        for subject in subjects:
            indices_per_subject = np.argwhere(indices == int(subject))
            X_per_subject = X[indices_per_subject[:, 0]]

            if args.dimension == 2:
                X_per_subject = X_per_subject.reshape(-1, 1, 2)
            elif args.dimension == 3:
                X_per_subject = X_per_subject.reshape(-1, 1, 3)

            segments = np.concatenate(
                [X_per_subject[:-1], X_per_subject[1:]], axis=1
            )

            if args.dimension == 2:
                instance = matplotlib.collections.LineCollection
            elif args.dimension == 3:
                instance = Line3DCollection

            lc = instance(
                segments,
                color='k',
                zorder=0,
                linewidths=1.0,
            )

            if args.dimension == 2:
                ax.add_collection(lc)
            elif args.dimension == 3:
                # TODO: until the issue with the line collection have
                # been fixed, do *not* show them for 3D plots.
                pass

        if encoder == 'm-phate':
            indices = np.repeat(np.arange(n_time_steps), n_samples)

        if args.dimension == 2:
            scatter = ax.scatter(
                X[:, 0], X[:, 1], c=indices, cmap='Spectral',
                zorder=10,
                s=10.0,
            )
        elif args.dimension == 3:
            scatter = ax.scatter(
                X[:, 0], X[:, 1], X[:, 2], c=indices, cmap='Spectral',
                zorder=10,
                s=10.0,
            )

        plt.colorbar(scatter)

        path = f'../../figures/persistence_images_embeddings/{basename}'

        plt.tight_layout()
        plt.savefig(
            os.path.join(path, f'{args.encoder}_global.png'),
            bbox_inches='tight'
        )

        plt.close(fig)

        if args.dimension == 2:

            fig, ax = plt.subplots()
            _, _, _, hist = ax.hist2d(
                                X[:, 0], X[:, 1],
                                bins=30,
                                cmap='viridis',
            )

            fig.colorbar(hist)

            plt.tight_layout()
            plt.savefig(
                os.path.join(path, f'{args.encoder}_density.png'),
                bbox_inches='tight'
            )

            plt.close(fig)

        refit = False
    else:
        refit = True

    # M-PHATE only supports global embeddings
    if args.encoder == 'm-phate':
        sys.exit(0)

    for subject in tqdm(subjects, desc='Subject'):
        embed(
            encoder,
            subject,
            data[subject],
            basename,
            prefix=args.encoder,
            metric=args.metric,
            rolling=args.rolling,
            refit=refit,
        )
