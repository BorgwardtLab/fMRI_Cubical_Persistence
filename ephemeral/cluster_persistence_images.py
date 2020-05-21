#!/usr/bin/env python3
#
# Cluster persistence images of participants and stores the linkage
# matrix and the distance matrix.

import argparse
import collections
import glob
import json
import os

import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
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

    args = parser.parse_args()

    with open(args.INPUT) as f:
        data = json.load(f)

    X = []

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
        images = np.array(data[subject])
        x = np.mean(images, axis=0).ravel()

        X.append(x)

    X = np.array(X)

    raise 'heck'

    # Use subject labels as 'true' labels (even though we have no way of
    # telling in a clustering setup)
    y = list(sorted(data.keys()))

    clf = AgglomerativeClustering(
        distance_threshold=0,      # compute full dendrogram
        n_clusters=None,           # do not limit number of clusters
        affinity='precomputed',    # use our distances from above
        linkage='average',         # cannot use Ward linkage here
    )
    model = clf.fit(D)
    M = get_linkage_matrix(model)

    experiment = os.path.basename(args.INPUT)
    experiment = os.path.splitext(args.INPUT)[0]

    # FIXME: needs to be made configurable; this presumes that we are
    # being called from another folder.
    os.chdir('../../results/clusterings')

    dimensions_str = '_'.join([str(d) for d in args.dimensions])
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

    output_filename = f'Assignments_{args.experiment}.json'

    with open(output_filename, 'w') as f:
        json.dump(assignments, f, indent=4)
