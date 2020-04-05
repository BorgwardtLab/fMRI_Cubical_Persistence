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

from tqdm import tqdm


def embed(subject, data, d, suffix):
    """Embed data of a given subject.

    Performs the embedding for a given subject and stores the resulting
    plot in an output directory.

    Parameters
    ----------
    subject : str
        Name of the subject to embed

    data : list of `np.array`
        List of high-dimensional vectors corresponding to the feature
        vectors at a given time step.

    d : int
        Embedding dimension

    suffix : str
        Suffix to use for storing embeddings
    """
    X = np.array([row for row in data])

    pca = PCA(n_components=d, random_state=42)
    X = pca.fit_transform(X)

    colours = np.linspace(0, 1, len(X))
    points = X.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = matplotlib.collections.LineCollection(segments, cmap='Spectral')
    lc.set_array(colours)

    plt.clf()

    min_x = X[:, 0].min()
    max_x = X[:, 0].max()
    min_y = X[:, 1].min()
    max_y = X[:, 1].max()

    min_ = min(min_x, min_y)
    max_ = max(max_x, max_y)

    plt.gca().add_collection(lc)
    plt.gca().set_xlim(min_, max_)
    plt.gca().set_ylim(min_, max_)
    plt.gca().set_aspect('equal')

    path = f'../figures/persim_embeddings/{suffix}'

    # Create output directories for storing *all* subjects in. This
    # depends on the input file.
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f'{subject}.png'))


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

    args = parser.parse_args()

    with open(args.INPUT) as f:
        data = json.load(f)

    basename = os.path.basename(args.INPUT)
    basename = os.path.splitext(basename)[0]

    subjects = data.keys()
    for subject in tqdm(subjects, desc='Subject'):
        embed(subject, data[subject], args.dimension, basename)
