#!/usr/bin/env python3
#
# Calculates persistence diagrams from (correlation) matrices instead of
# raw volumes. Technically, this could also be solved with DIPHA, but in
# our case, doing it directly on a graph is also possible.

import argparse

import igraph as ig
import numpy as np

from pyper import persistent_homology
from utilities import parse_filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', type=str, help='Input file(s)')

    args = parser.parse_args()

    for filename in args.INPUT:
        X = np.load(filename, allow_pickle=True)
        X = X['X']

        # Required for book-keeping and storing a persistence diagram
        # later on.
        subject, _, _ = parse_filename(filename)

        # Create a graph from the vertex representation; technically,
        # this should be the full graph, but we reserve the right to
        # remove some edges if they are deemed to be non-existent.
        G = ig.Graph.Adjacency((X != 0).tolist())

        # Use the correlations as edge weights and assign the minimum
        # value to all vertices.
        G.es['weight'] = X[X.nonzero()]
        G.vs['weight'] = np.min(X)

        pd_0, pd_1 = persistent_homology.calculate_persistence_diagrams(
                graph=G,
                vertex_attribute='weight',
                edge_attribute='weight',
                order='sublevel'
        )
