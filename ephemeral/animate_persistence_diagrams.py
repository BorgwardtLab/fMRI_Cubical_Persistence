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


def animate_persistence_diagram_sequence(
    filenames: list,
    output: str = None,
    dimension: int = 2
):
    """Animate a sequence of persistence diagrams.

    This function animates a sequence of persistence diagrams by
    plotting them in a repeating sequence. The animation is either shown
    directly or stored in a file.

    Parameters
    ----------

    filenames:
        List of input filenames

    output:
        Output filename (optional). If set to `None`, the function will
        just open a plot on the local machine.

    dimension:
        Dimension to plot
    """
    fig = plt.figure(figsize=(10, 10))

    min_c = float('inf')
    min_d = float('inf')
    max_c = -float('inf')
    max_d = -float('inf')

    title = ''

    # Stores coordinates for each time step. The keys are the time
    # steps, while the values are dictionaries containing coordinates
    # for each persistence diagram.
    coords_per_timestep = collections.defaultdict(dict)

    for filename in tqdm(filenames, desc='Filename'):
        dims, creation, destruction = load_persistence_diagram_json(
                                        filename
        )

        subject, _, time = parse_filename(filename)

        if not title:
            title = subject

        c = creation[dims == dimension].ravel()
        d = destruction[dims == dimension].ravel()

        coords_per_timestep[time]['x'] = c.tolist()
        coords_per_timestep[time]['y'] = d.tolist()

        min_c = min(min_c, np.min(c))
        max_c = max(max_c, np.max(c))
        min_d = min(min_d, np.min(d))
        max_d = max(max_d, np.max(d))

    plt.suptitle(f'Subject: {subject}')

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

    animate_persistence_diagram_sequence(
        args.FILES,
        dimensions=args.dimensions
    )
