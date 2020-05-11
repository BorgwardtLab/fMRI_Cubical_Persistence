#!/usr/bin/env python3
#
# Performs clustering evaluation of a given linkage matrix, using the
# set of pre-defined labels.

import argparse

import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('DISTANCES', help='Path to a distance matrix.')

    parser.add_argument(
        '-c', '-clusters',
        type=int,
        help='Specifies the number of clusters to use.'
    )

    # TODO: does it make sense to make this configurable or can we
    # 'guess' it directly from the path to the linkage matrix?
    parser.add_argument(
        '-l', '--labels',
        type=str,
        default='../results/clusterings/Labels.txt',
        help='Specifies path to cluster labels.'
    )

    args = parser.parse_args()

    # Contains more than just the cluster assignments; this data frame
    # also has a column for the participant label itself.
    y_true = pd.read_csv('../data/participant_groups.csv')

    n_groups = len(y_true['cluster'].unique())
    unique_groups = sorted(y_true['cluster'].unique())

    distances = np.loadtxt(args.DISTANCES)

    # Required to perform recursive splits of clusters; this is done in
    # order to measure the agreement between actual labels and predicted
    # labels, while maintaining clusters of roughly equal size.
    D = distances.copy()

    clf = AgglomerativeClustering(
        affinity='precomputed',
        linkage='average',
        n_clusters=2,
    )

    for i in range(2, n_groups + 1):

        best_score = None
        best_k = None

        # Perform a binary split
        for k in unique_groups:
            y = y_true['cluster'].to_numpy(copy=True)
            y[y != k] = -1
            y[y == k] = 1

            y_pred = clf.fit_predict(D)
            score = adjusted_rand_score(y, y_pred)

            if best_score is None:
                best_score = score
                best_k = k
            elif score > best_score:
                best_score = score
                best_k = k

        print('best score =', best_score)
        print('best k =', best_k)
