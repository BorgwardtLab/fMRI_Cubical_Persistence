#!/usr/bin/env python3
#
# Simple temporal coherence analysis of persistence images. At first,
# only a simple measure will be tried, viz. to what extent neighbours
# of each point are more than $k$ steps apart.

import argparse
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-t', type=int, default=None)

    args = parser.parse_args()

    if args.t is None:
        args.t = args.k

    coherence_values = []

    with open(args.INPUT) as f:
        data = json.load(f)

    for subject in tqdm(sorted(data.keys()), desc='Subject'):

        # Check whether we are dealing with a proper subject, or an
        # information key in the data.
        try:
            _ = int(subject)
        except ValueError:
            continue

        X = np.array(data[subject])

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
