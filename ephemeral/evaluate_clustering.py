#!/usr/bin/env python3
#
# Performs clustering evaluation of a given linkage matrix, using the
# set of pre-defined labels.

import argparse
import os

import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score


def evaluate_global_clustering(D, k, y_true):
    """Evaluate global clustering of full distance matrix."""
    clf = AgglomerativeClustering(
        affinity='precomputed',
        linkage='average',
        n_clusters=k,
    )

    y_pred = clf.fit_predict(D)

    print('Global analysis')
    print('Global predictions:', y_pred)
    print('AMI:',  adjusted_mutual_info_score(y_true, y_pred))
    print('ARI:',  adjusted_rand_score(y_true, y_pred))


def evaluate_local_clustering(distances, Y):
    """Evaluate local clustering of subset distance matrix."""
    # Required to perform recursive splits of clusters; this is done in
    # order to measure the agreement between actual labels and predicted
    # labels, while maintaining clusters of roughly equal size.
    D = distances.copy()

    clf = AgglomerativeClustering(
        affinity='precomputed',
        linkage='average',
        n_clusters=2,
    )

    # Keeps track of the true cluster assignments and the 'synthetic'
    # assignments generated by the algorithm below.
    y_true = Y['cluster'].to_numpy()
    y_pred_synthetic = np.full(y_true.shape, -1)

    original_indices = np.arange(len(y_true))

    # Contains the 'best' value of $k$ for each split. This permits
    # a merge strategy later on.
    best_k_values = []

    for i in range(2, n_groups + 1):

        best_prediction = None
        best_score = None
        best_k = None

        # Perform a binary split of the 'remaining' labels; the pool of
        # available labels gets smaller and smaller, as we are removing
        # the items for which we already have a prediction.
        for k in unique_groups:
            y = np.array(y_true)
            y[y != k] = -1
            y[y == k] = 1

            y_pred = clf.fit_predict(D)
            score = adjusted_mutual_info_score(y, y_pred)

            if best_score is None:
                best_prediction = y_pred
                best_score = score
                best_k = k
            elif score >= best_score:
                best_prediction = y_pred
                best_score = score
                best_k = k

        if args.show_scores:
            print('best prediction =', best_prediction)
            print('best score =', best_score)
            print('best k =', best_k)

        best_k_values.append(best_k)

        counts = np.bincount(best_prediction)
        smaller_group = np.argmin(counts)

        # Remove the predictions under the 'best' clustering identified
        # above.
        indices, = np.nonzero(best_prediction != smaller_group)
        D = D[indices, :]
        D = D[:, indices]

        # Ditto for the 'true' labels (note that we could remove some
        # incorrect labels here).
        y_true = y_true[indices]

        # Create synthetic predictions by re-using the original label
        # from above. Make sure that this vector uses the *original*,
        # i.e. the ones pertaining to `D`, indices.
        original_indices = original_indices[indices]
        y_pred_synthetic[~original_indices] = best_k

        # Not necessary, strictly speaking, but better style. Ensures
        # that we never perform an empty fit.
        unique_groups.remove(best_k)

    # Ensures that all groups are being accounted for, both in the
    # hierarchy *and* in the predictions.
    assert len(unique_groups) == 1
    best_k_values.append(unique_groups[0])
    y_pred_synthetic[y_pred_synthetic == -1] = unique_groups[0]

    # Reset the original labels
    y_true = Y['cluster'].to_numpy()

    print('Local analysis')
    print('Suggested cluster similarity order:', best_k_values)
    print('Synthetic predictions:', y_pred_synthetic)
    print('AMI:',  adjusted_mutual_info_score(y_true, y_pred_synthetic))
    print('ARI:',  adjusted_rand_score(y_true, y_pred_synthetic))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'DISTANCES',
        type=str,
        nargs='+',
        help='Path to a distance matrix.'
    )

    # TODO: does it make sense to make this configurable or can we
    # 'guess' it directly from the path to the linkage matrix?
    parser.add_argument(
        '-l', '--labels',
        type=str,
        default='../results/clusterings/Labels.txt',
        help='Specifies path to cluster labels.'
    )

    parser.add_argument(
        '-s', '--show-scores',
        action='store_true',
        help='If set, shows scores per iteration'
    )

    args = parser.parse_args()

    # Contains more than just the cluster assignments; this data frame
    # also has a column for the participant label itself.
    Y = pd.read_csv('../data/participant_groups.csv')

    n_groups = len(Y['cluster'].unique())
    unique_groups = sorted(Y['cluster'].unique())

    for filename in args.DISTANCES:
        print('-' * 72)
        print(os.path.splitext(os.path.basename(filename))[0])
        print('-' * 72)

        distances = np.loadtxt(filename)

        # This checks how well we can approximate the full clustering on
        # a global level.
        evaluate_global_clustering(
            distances,
            n_groups,
            Y['cluster']
        )

        evaluate_local_clustering(
            distances,
            Y
        )
