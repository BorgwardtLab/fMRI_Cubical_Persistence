#!/usr/bin/env python3

import argparse
import collections

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from topology import load_persistence_diagram_json
from utilities import parse_filename

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

    # Prepare figure and empty scatterplot to prevent flickering in the
    # animation.
    fig, ax = plt.subplots(figsize=(5, 5))
    scatter = ax.scatter([], [])

    def _init_fn():
        ax.set_xlim(min_c, max_c)
        ax.set_ylim(min_d, max_d)

    def _update_fn(frame):
        x = coords_per_timestep[frame]['x']
        y = coords_per_timestep[frame]['y']

        # Despite the name, this updates the *positions* of the points
        # in the scatter plot.
        data = np.column_stack((x, y))
        scatter.set_offsets(data)

        fig.suptitle(f'Subject: {subject}, d = {dimension}, t = {frame}')

    time_steps = sorted(coords_per_timestep.keys())

    ani = animation.FuncAnimation(
        fig,
        _update_fn,
        frames=time_steps,
        init_func=_init_fn,
        interval=300
    )

    if output is not None:
        ani.save(output, dpi=300, writer='imagemagick')
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', nargs='+', help='Input file(s)')
    parser.add_argument(
        '-d',
        '--dimension',
        type=int,
        default=2,
        help='Indicates which dimension to plot'
    )

    parser.add_argument(
        '-o',
        '--output',
        default=None,
        help='Output file'
    )

    args = parser.parse_args()

    animate_persistence_diagram_sequence(
        args.FILES,
        dimension=args.dimension,
        output=args.output
    )
