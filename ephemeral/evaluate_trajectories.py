#!/usr/bin/env python3
#
# Simple temporal coherence analysis of trajectories. A trajectory is
# temporally coherent if the $k$ nearest neighbours of each point are
# not more than $k$ steps apart.

import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import NearestNeighbors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', type=str, help='Input file(s)')
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-t', type=int, default=None)

    args = parser.parse_args()

    if args.t is None:
        args.t = args.k

    incoherency_values = []

    for filename in args.INPUT:
        df = pd.read_csv(filename, index_col='time')
        X = df[['x', 'y']].to_numpy()

        nn = NearestNeighbors(n_neighbors=args.k)
        nn.fit(X)

        all_neighbours = nn.kneighbors(
            X, n_neighbors=args.k, return_distance=False
        )

        n_incoherent_points = 0

        for index, neighbours in enumerate(all_neighbours):
            is_incoherent = (neighbours > index + args.t).sum() \
                          + (neighbours < index - args.t).sum()

            is_incoherent /= args.k

            n_incoherent_points += is_incoherent

        incoherency_values.append(n_incoherent_points / len(X) * 100)

    sns.distplot(incoherency_values, bins=20)
    plt.show()
