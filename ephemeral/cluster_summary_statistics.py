#!/usr/bin/env python3
#
# Cluster summary statistics curves, storing the linkage and the
# distance matrix the distance matrix, as well as depicting the
# embedding.

import argparse
import json
import os

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

from tqdm import tqdm


def get_linkage_matrix(model, **kwargs):
    """Calculate linkage matrix and return it."""
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    return linkage_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')

    parser.add_argument(
        '-s', '--statistic',
        type=str,
        help='Summary statistic to extract',
        required=True,
    )

    args = parser.parse_args()

    with open(args.INPUT) as f:
        data = json.load(f)

    df = []

    for subject in tqdm(sorted(data.keys()), desc='Subject'):

        # Check whether we are dealing with a proper subject, or an
        # information key in the data.
        try:
            _ = int(subject)
        except ValueError:
            continue

        # Store a mean representation of the persistence image of each
        # participant. This is not necessarily the smartest choice but
        # a very simple one.
        curve = data[subject][args.statistic]
        df.append(curve)

    df = pd.DataFrame(df)
    df_cohorts = pd.read_csv('../data/participant_groups.csv')
    df['cohort'] = df_cohorts['cluster']
    df['cohort'] = df['cohort'].astype('str')

    def _normalise_cohort(df):
        df = df.select_dtypes(np.number)
        df = (df - df.min()) / (df.max() - df.min())
        return df

    # Make the cohort curves configurable; since we are only showing
    # relative variabilities, this is justified.
    df.loc[:, df.columns.drop('cohort')] = df.groupby('cohort').apply(
        _normalise_cohort
    )

    X = df.select_dtypes(np.number).to_numpy()

    df_ages = pd.read_csv('../data/participant_ages.csv')
    df['age'] = df_ages['Age']

    #X = X[df['age'] < 18]
    #df = df[df['age'] < 18]

    D = pairwise_distances(X, metric='l1')
    Y = MDS(dissimilarity='precomputed', metric=True).fit_transform(D)

    plt.scatter(
        x=Y[:, 0],
        y=Y[:, 1],
        c=df['age'].values,
    )

    plt.colorbar()
    plt.show()

    clf = AgglomerativeClustering(
        distance_threshold=0,      # compute full dendrogram
        n_clusters=None,           # do not limit number of clusters
        affinity='precomputed',    # use our distances from above
        linkage='average',         # cannot use Ward linkage here
    )
    model = clf.fit(D)
    M = get_linkage_matrix(model)

    raise 'heck'

    experiment = os.path.basename(args.INPUT)
    experiment = os.path.splitext(experiment)[0]

    # FIXME: needs to be made configurable; this presumes that we are
    # being called from another folder.
    os.chdir('../results/clusterings')

    out_filename = f'Linkage_matrix_{experiment}.txt'

    np.savetxt(out_filename, M)
    np.savetxt('Labels.txt', y, fmt='%s')

    # Also store the distance matrix. This makes it possible not only to
    # completely reconstruct the clustering but also try out different
    # settings for each algorithm later on.
    out_filename = f'Distance_matrix_{experiment}.txt'

    np.savetxt(out_filename, D)

    # Get cluster assignments for simple binary clustering; this is the
    # easiest clustering we can do here.
    clf = AgglomerativeClustering(affinity='precomputed', linkage='average')
    y_pred = clf.fit_predict(D).tolist()

    assignments = {
        subject: label for (subject, label) in zip(y, y_pred)
    }

    output_filename = f'Assignments_{experiment}.json'

    with open(output_filename, 'w') as f:
        json.dump(assignments, f, indent=4)
