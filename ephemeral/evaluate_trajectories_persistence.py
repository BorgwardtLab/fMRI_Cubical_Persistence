#!/usr/bin/env python3
#
# Simple persistence-based analysis of trajectories. This script
# requires a working installation of Aleph. More precisely, the
# tool `vietoris_rips` must be available in the path.

import argparse
import io

import numpy as np
import pandas as pd

from topology import PersistenceDiagram

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import pairwise_distances

from subprocess import check_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', type=str, help='Input file(s)')

    args = parser.parse_args()

    trajectory_information = []

    for filename in args.INPUT:

        print(filename)

        df = pd.read_csv(filename, index_col='time')
        X = df[['x', 'y']].to_numpy()

        min_coordinate = np.min(np.min(X, axis=0))
        max_coordinate = np.max(np.max(X, axis=0))

        X = (X - min_coordinate) / (max_coordinate - min_coordinate)

        np.savetxt('/tmp/foo.txt', X, fmt='%.8f')

        output = check_output(
                ['vietoris_rips', '-t', '/tmp/foo.txt', '0.05', '2'],
                universal_newlines=True,
        )

        output = io.StringIO(output)
        Y = np.genfromtxt(output)

        # Ensures that we always obtain a 2D array even if only a single
        # cycle is present.
        if len(Y.shape) == 1:
            Y = Y.reshape(1, -1)

        # This makes it easier to quantify the topological information
        # in the trajectory.
        D = PersistenceDiagram(
            dimension=2,
            creation_values=Y[:, 0],
            destruction_values=Y[:, 1]
        )

        trajectory_information.append(
            {
                'infinity_norm_p1': D.infinity_norm(1.0),
                'infinity_norm_p2': D.infinity_norm(2.0),
                'total_persistence_p1': D.total_persistence(1.0),
                'total_persistence_p2': D.total_persistence(2.0),
            }
        )

    df = pd.DataFrame.from_dict(trajectory_information)
    print(df)
