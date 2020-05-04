#!/usr/bin/env python3
#
# Visualises persistence-based embeddings in the form of trajectories
# in a space of a given dimension.

import argparse
import json
import os
import sys

import matplotlib.collections
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Line3DCollection

import numpy as np

from sklearn.decomposition import PCA

from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

from sklearn.preprocessing import StandardScaler

from phate import PHATE
from m_phate import M_PHATE

from tqdm import tqdm


def embed(
    encoder,
    subject,
    data,
    suffix,
    prefix=None,
    metric=None,
    refit=True
):
    """Embed data of a given subject.

    Performs the embedding for a given subject and stores the resulting
    plot in an output directory.

    Parameters
    ----------
    encoder : `TransformerMixin`
        Transformer class for performing the embedding or encoding of
        the data.

    subject : str
        Name of the subject to embed

    data : list of `np.array`
        List of high-dimensional vectors corresponding to the feature
        vectors at a given time step.

    suffix : str
        Suffix to use for storing embeddings

    prefix : str, or `None` (optional)
        Prefix to use for naming each individual embedding. Can be used
        to distinguish between different sorts of outputs.

    metric : str, callable, or `None` (optional)
        If set, specifies the distance metric to use for the embedding
        process. Needs to be a metric recognised by `scikit-learn`, or
        a callable function.

    refit : bool
        If set, refits the encoder to the current data set. Else, the
        encoder is used as-is.
    """
    X = np.array([row for row in data])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # TODO: decide whether this will be useful or not
    # X -= np.mean(X, axis=0)

    if metric is not None:
        X = pairwise_distances(X, metric=metric)

        # TODO: check whether the classifier supports this
        encoder.set_params(dissimilarity='precomputed')

    if refit:
        X = encoder.fit_transform(X)
    else:
        X = encoder.transform(X)

    colours = np.linspace(0, 1, len(X))
    points = X.reshape(-1, 1, args.dimension)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if args.dimension == 2:
        instance = matplotlib.collections.LineCollection
    elif args.dimension == 3:
        instance = Line3DCollection

    lc = instance(segments, cmap='Spectral')
    lc.set_array(colours)

    if args.dimension == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots()

    fig.set_size_inches(5, 5)

    min_x = X[:, 0].min()
    max_x = X[:, 0].max()
    min_y = X[:, 1].min()
    max_y = X[:, 1].max()

    if args.dimension == 3:
        ax.add_collection3d(lc)

        min_z = X[:, 2].min()
        max_z = X[:, 2].max()

        ax.set_zlim(min_z, max_z)

    elif args.dimension == 2:
        ax.add_collection(lc)

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    path = f'../figures/persistence_images_embeddings/{suffix}'

    # Create output directories for storing *all* subjects in. This
    # depends on the input file.
    os.makedirs(path, exist_ok=True)
    plt.tight_layout()

    name = ''

    if prefix is not None:
        name += f'{prefix}'

    # TODO: this cannot handle callable arguments yet, but at least some
    # simple defaults.
    if type(metric) is str:
        name += f'_{metric}'

    name += f'_{subject}'

    plt.savefig(
        os.path.join(path, f'{name}.png'),
        bbox_inches='tight'
    )

    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'INPUT',
        help='Input file (JSON)',
        type=str
    )
    parser.add_argument(
        '-d', '--dimension',
        help='Dimension of visualisation',
        type=int,
        default=2,
    )

    parser.add_argument(
        '-e', '--encoder',
        help='Specifies encoding/embedding method',
        type=str,
        default='pca'
    )

    parser.add_argument(
        '-m', '--metric',
        help='Specifies metric for calculating embedding',
        type=str,
    )

    parser.add_argument(
        '-g', '--global-embedding',
        action='store_true',
        help='If set, calculates *global* embeddings'
    )

    parser.add_argument(
        '-t', '--trajectories',
        action='store_true',
        help='If set, shows trajectory visualisations'
    )

    args = parser.parse_args()

    with open(args.INPUT) as f:
        data = json.load(f)

    basename = os.path.basename(args.INPUT)
    basename = os.path.splitext(basename)[0]

    if args.encoder == 'pca':
        encoder = PCA(
            n_components=args.dimension,
            random_state=42
        )
    elif args.encoder == 'mds':
        encoder = MDS(
            n_components=args.dimension,
            random_state=42,
        )
    elif args.encoder == 'tsne':
        encoder = TSNE(
            n_components=args.dimension,
            random_state=42,
        )
    elif args.encoder == 'lle':
        encoder = LocallyLinearEmbedding(
            n_components=args.dimension,
            random_state=42,
        )
    elif args.encoder == 'phate':
        encoder = PHATE(
            n_components=args.dimension,
            random_state=42,
        )
    elif args.encoder == 'm-phate':
        encoder = M_PHATE(
            n_components=args.dimension,
            random_state=42
        )

    # Filter subjects; this could be solved smarter...
    subjects = data.keys()
    subjects = [subject for subject in subjects if len(subject) == 3]

    if args.global_embedding:
        if args.dimension == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots()

        # Fit the estimator on *all* subjects first, then attempt to
        # embed them individually.
        X = np.concatenate(
            [np.array(data[subject]) for subject in subjects]
        )

        # Create array full of subject indices. We can use this to
        # colour-code subjects afterwards.
        indices = np.concatenate([
            np.full(fill_value=int(subject),
                    shape=np.array(data[subject]).shape[0])
            for subject in subjects
        ])

        X = encoder.fit_transform(X)

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
                ax.add_collection3d(lc)

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

        path = f'../figures/persistence_images_embeddings/{basename}'

        plt.colorbar(scatter)
        plt.tight_layout()
        plt.savefig(os.path.join(path, f'{args.encoder}_global.png'))

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
            refit=refit,
        )
