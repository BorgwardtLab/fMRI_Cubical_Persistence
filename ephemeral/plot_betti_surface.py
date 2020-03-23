#!/usr/bin/env python3
#
# Script for calculating a Betti surface, i.e. a time-varying Betti
# curve, for a test subject.

import argparse
import collections

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utilities import load_graphs

from topology import load_persistence_diagram_json
from topology import make_betti_curve

from utilities import parse_filename

from tqdm import tqdm


def make_betti_surface(betti_curves):
    """Create Betti surface and return it."""
    # Merge the individual Betti curves of each single channel to one
    # Betti surface.

    index = pd.Index([])
    for betti_curve in betti_curves:
        index = index.union(betti_curve._data.index)

    index = index.drop_duplicates(keep='first')
    curves = [
        curve._data.reindex(index, method='ffill') for curve in betti_curves
    ]

    # This adds all time steps manually to the curves, which is all
    # right since we do not have that information available anyway.
    columns = {
        time: curve for time, curve in enumerate(curves)
    }

    betti_surface = pd.DataFrame(data=columns, index=index)
    return betti_surface


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', nargs='+', help='Input files')
    parser.add_argument(
        '-d',
        '--dimensions',
        type=list,
        default=[2],
        help='Indicates which dimensions to use'
    )

    args = parser.parse_args()

    subject = None
    betti_curves = []

    for filename in tqdm(args.FILES, desc='File'):
        sub, _, _time = parse_filename(filename)

        # Make sure that we are loading diagrams from a single subject
        # only (and only from the same one).
        if subject is None:
            subject = sub
        else:
            assert subject == sub

        persistence_diagrams = load_persistence_diagram_json(
            filename, return_raw=False
        )

        # Remove diagrams that are unnecessary
        persistence_diagrams = [
            D for D in persistence_diagrams if D.dimension in args.dimensions
        ]

        # Create all Betti curves for the set of diagrams. This is
        # slightly stupid and does not account for different dimensions
        # so far.
        betti_curves_ = [
            make_betti_curve(diagram) for D in persistence_diagrams
        ]

        betti_curves.extend(betti_curves_)

    betti_surface = make_betti_surface(betti_curves)

    graphs = load_graphs(args.INPUT)

    # Stores all Betti curves for the individual time steps. The key of
    # this collection is the channel number of the curve. This is zero,
    # except for when multiple curves are being calculated (for another
    # filtration like the height filtration).
    curves = collections.defaultdict(list)

    for index, graph in enumerate(graphs):

        # For the distance-based filtration, everything is easy: we just
        # calculate *one* filtration and get *one* graph.
        if args.filtration == 'distance':
            graph = calculate_distance_filtration(graph)

            pd_0, _ = calculate_persistence_diagrams(graph)
            betti_curve = make_betti_curve(pd_0)

            # TODO: better access of the series data? Maybe the
            # object-oriented interface is not required.
            curves[0].append(betti_curve._data)

        # With the height filtration, it is more complicated: choose
        # different 'viewpoints' on the sphere and create *multiple*
        # filtrations, resulting in *multiple* graphs.
        elif args.filtration == 'height':

            # TODO: make these directions selectable?
            directions = [
                [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
            ]

            for j, direction in enumerate(directions):
                graph = calculate_height_filtration(graph, direction)

                pd_0, _ = calculate_persistence_diagrams(graph)
                betti_curve = make_betti_curve(pd_0)

                curves[j].append(betti_curve._data)

    n_directions = len(curves.keys())
    fig, ax = plt.subplots(n_directions)

    # Merge the individual Betti curves of each single channel to one
    # Betti surface.
    for key, curves_ in curves.items():

        index = pd.Index([])
        for curve in curves_:
            index = index.union(curve.index)

        index = index.drop_duplicates(keep='first')

        # Forward-filling of all curves is allowed because the function, by
        # definition, *cannot* change outside the defined thresholds.
        curves_ = [curve.reindex(index, method='ffill') for curve in curves_]

        # This adds all time steps manually to the curves, which is all
        # right since we do not have that information available anyway.
        columns = {
            time: curve for time, curve in enumerate(curves_)
        }

        betti_surface = pd.DataFrame(data=columns, index=index)

        if n_directions != 1:
            sns.heatmap(betti_surface, yticklabels=False, ax=ax[key])
        else:
            sns.heatmap(betti_surface, yticklabels=False, ax=ax)



    plt.show()

    # HERE BE MORE DRAGONS

    sns.set_style('darkgrid')

    # Check whether we need subplots or not
    dimension = data.shape[1] - 1

    # The boring case: only a single summary statistic. Nothing to do
    # here.
    if dimension == 1:
        sns.lineplot(data[:, 0], data[:, 1])
    else:

        for d in range(dimension):
            sns.lineplot(data[:, 0], data[:, d+1], ax=ax[d])

    plt.show()
