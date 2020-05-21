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

    coherence_values = []

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

        coherence_values.append(100 - n_incoherent_points / len(X) * 100)

    df = pd.read_csv('../data/participant_groups.csv')
    df['coherence'] = coherence_values

    print(df.groupby('cluster')['coherence'].agg(['mean', 'std']))

    print(df[df['cluster'] != 5].agg(['mean', 'std']))

    ax = sns.boxplot(x='cluster', y='coherence', data=df)
    ax = sns.swarmplot(
            x='cluster', y='coherence',
            data=df,
            ax=ax,
            color='.25'
        )

    plt.show()
