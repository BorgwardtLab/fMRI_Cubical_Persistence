#!/usr/bin/env python3

import argparse

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
    fig = plt.figure(figsize=(40, 80))

    for index, dimension in enumerate(dimensions):

        # Minima/maxima for the respective sequence of patients
        min_c = float('inf')
        min_d = float('inf')
        max_c = -float('inf')
        max_d = -float('inf')

        for filename in tqdm(filenames, desc='Filename'):
            dims, creation, destruction = load_persistence_diagram_json(
                                            filename
            )

            subject, _, time = parse_filename(filename)

            ax = fig.add_subplot(
                    1,
                    len(dimensions),
                    index + 1,
                    projection='3d'
                )

            ax.set_title(f'Dimension: {dimension}')

            c = creation[dims == dimension].ravel()
            d = destruction[dims == dimension].ravel()
            t = [float(time)] * len(c)

            min_c = min(min_c, np.min(c))
            max_c = max(max_c, np.max(c))
            min_d = min(min_d, np.min(d))
            max_d = max(max_d, np.max(d))

            ax.scatter(c, t, d)

            ax.set_xlabel('Creation')
            ax.set_ylabel('$t$')
            ax.set_zlabel('Destruction')

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
