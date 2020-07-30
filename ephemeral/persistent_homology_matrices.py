#!/usr/bin/env python3
#
# Calculates persistence diagrams from (correlation) matrices instead of
# raw volumes. Technically, this could also be solved with DIPHA, but in
# our case, doing it directly on a graph is also possible.

import argparse
import json
import os

import igraph as ig
import numpy as np

from pyper import persistent_homology

from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', type=str, help='Input file(s)')
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output directory',
        required=True
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for filename in tqdm(args.INPUT, desc='File'):
        X = np.load(filename, allow_pickle=True)
        X = X['X']

        # Create a graph from the vertex representation; technically,
        # this should be the full graph, but we reserve the right to
        # remove some edges if they are deemed to be non-existent.
        G = ig.Graph.Adjacency((X != 0).tolist())

        # Use the transformed correlations as edge weights and assign
        # the minimum distance value to all vertices.
        G.es['weight'] = 1 - X[X.nonzero()]
        G.vs['weight'] = 0.0

        pd_0, _ = persistent_homology.calculate_persistence_diagrams(
                graph=G,
                vertex_attribute='weight',
                edge_attribute='weight',
                order='sublevel'
        )

        pairs = np.asarray(pd_0._pairs)

        dimensions = [0] * len(pairs)
        creation_values = pairs[:, 0].tolist()
        destruction_values = pairs[:, 1].tolist()

        assert len(dimensions) == len(creation_values)

        basename = os.path.basename(filename)
        basename = os.path.splitext(basename)[0]
        basename = basename + '.json'

        with open(os.path.join(args.output, basename), 'w') as f:
            json.dump(
                {
                    'dimensions': dimensions,
                    'creation_values': creation_values,
                    'destruction_values': destruction_values
                },
                f, indent=4
            )
