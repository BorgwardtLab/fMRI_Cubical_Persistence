#!/usr/bin/env python3
#
# Visualises persistence-based embeddings in the form of trajectories
# in a space of a given dimension.

import argparse
import json
import os

import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA

from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

from tqdm import tqdm


def embed(encoder, subject, data, suffix, prefix=None, metric=None):
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
    """
    X = np.array([row for row in data])

    if metric is not None:
        X = pairwise_distances(X, metric=metric)

    X = encoder.fit_transform(X)

    colours = np.linspace(0, 1, len(X))
    points = X.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = matplotlib.collections.LineCollection(segments, cmap='Spectral')
    lc.set_array(colours)

    plt.clf()
    plt.gcf().set_size_inches(5, 5)

    min_x = X[:, 0].min()
    max_x = X[:, 0].max()
    min_y = X[:, 1].min()
    max_y = X[:, 1].max()

    plt.gca().add_collection(lc)
    plt.gca().set_xlim(min_x, max_x)
    plt.gca().set_ylim(min_y, max_y)
    plt.gca().set_aspect('equal')

    path = f'../figures/persim_embeddings/{suffix}'

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

    args = parser.parse_args()

    with open(args.INPUT) as f:
        data = json.load(f)

    basename = os.path.basename(args.INPUT)
    basename = os.path.splitext(basename)[0]

    if args.encoder == 'pca':
        encoder = PCA(n_components=args.dimension, random_state=42)
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

    subjects = data.keys()
    for subject in tqdm(subjects, desc='Subject'):
        embed(
            encoder,
            subject,
            data[subject],
            basename,
            prefix=args.encoder,
            metric=args.metric
        )
