#!/usr/bin/env python3
#
# Script for calculating a Betti surface, i.e. a time-varying Betti
# curve, for a test subject.

import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
            make_betti_curve(D) for D in persistence_diagrams
        ]

        betti_curves.extend(betti_curves_)

    betti_surface = make_betti_surface(betti_curves)


    fig, ax = plt.subplots()
    sns.heatmap(betti_surface, yticklabels=False, ax=ax)

    plt.savefig('Betti_curve.png')
