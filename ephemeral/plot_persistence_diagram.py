#!/usr/bin/env python3

import argparse
import collections

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from topology import load_persistence_diagram_json
from utilities import parse_filename

from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm


def plot_persistence_diagram_sequence(
    filenames: list,
    output: str = None,
    dimensions: list = [0, 1, 2],
):
    """Plot a sequence of persistence diagrams.

    This function plots a sequence of persistence diagram to a file or
    opens a visualisation directly. The persistence diagrams need to
    belong to the same subject, but this is not checked by the function.

    Parameters
    ----------

    filenames:
        List of input filenames

    output:
        Output filename (optional). If set to `None`, the function will
        just open a plot on the local machine.

    dimensions:
        List of dimensions to plot.

    task:
        Determines which persistence diagrams to load. Should not be
        necessary to change for now.
    """
    fig = plt.figure(figsize=(20, 8))

    # Minima/maxima for the respective sequence of patients
    min_c = float('inf')
    min_d = float('inf')
    max_c = -float('inf')
    max_d = -float('inf')

    # Stores coordinates for the 3D scatterplots; else, everything
    # else will be overwritten.
    coords_per_dimension = {
        dim: collections.defaultdict(list) for dim in dimensions
    }

    title = ''

    for filename in tqdm(filenames, desc='Filename'):
        dims, creation, destruction = load_persistence_diagram_json(
                                        filename
        )

        subject, _, time = parse_filename(filename)

        if not title:
            title = subject

        for dimension in dimensions:

            c = creation[dims == dimension].ravel()
            d = destruction[dims == dimension].ravel()
            t = [float(time)] * len(c)

            coords_per_dimension[dimension]['x'] += c.tolist()
            coords_per_dimension[dimension]['y'] += t
            coords_per_dimension[dimension]['z'] += d.tolist()

            min_c = min(min_c, np.min(c))
            max_c = max(max_c, np.max(c))
            min_d = min(min_d, np.min(d))
            max_d = max(max_d, np.max(d))

    plt.set_cmap('Spectral_r')
    plt.suptitle(f'Subject: {subject}')

    for index, dimension in enumerate(dimensions):

        ax = fig.add_subplot(
                1,
                len(dimensions),
                index + 1,
                projection='3d'
            )

        ax.set_title(f'Dimension: {dimension}')

        ax.set_xlabel('Creation')
        ax.set_ylabel('$t$')
        ax.set_zlabel('Destruction')

        ax.view_init(17, -60)
        ax.scatter(
            coords_per_dimension[dimension]['x'][::-1],
            coords_per_dimension[dimension]['y'][::-1],
            coords_per_dimension[dimension]['z'][::-1],
            c=coords_per_dimension[dimension]['y'],
            alpha=0.8,
            s=8,
        )

        ax.set_xlim(min_c, max_c)
        ax.set_zlim(min_d, max_d)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', nargs='+', help='Input file(s)')
    parser.add_argument(
        '-d',
        '--dimensions',
        nargs='+',
        type=int,
        default=[0, 1, 2],
        help='List indicating which dimensions to plot'
    )

    args = parser.parse_args()

    plot_persistence_diagram_sequence(
        args.FILES,
        dimensions=args.dimensions
    )
