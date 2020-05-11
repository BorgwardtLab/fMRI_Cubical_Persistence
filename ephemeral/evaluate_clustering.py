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

    distances = np.loadtxt(args.MATRIX)

    clf = AgglomerativeClustering(affinity='precomputed', linkage='average')
    y_pred = clf.fit_predict(distances)

    print(y_pred)
