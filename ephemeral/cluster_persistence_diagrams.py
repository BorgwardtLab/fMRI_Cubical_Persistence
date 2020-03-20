#!/usr/bin/env python3

import argparse
import collections
import glob
import json
import itertools
import os

import numpy as np

from features import featurise_distances
from topology import load_persistence_diagram_json
from utilities import parse_filename

from sklearn.cluster import AgglomerativeClustering

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


def vectorise_diagram(diagram, method):
    """Vectorise persistence diagram.

    Performs persistence diagram vectorisation based on a given method
    and returns a feature vector.

    Parameters
    ----------
    diagram : topology.PersistenceDiagram
        Persistence diagram to vectorise

    method : str
        Method for vectorisation. Can be either one of 'summary_statistics'
        or 'top_persistence'. This determines *how* features are stored.

    Returns
    -------
    Feature vector corresponding to the current diagram. Can be appended
    to form large-scale feature vectors, if desired.
    """
    if method == 'summary_statistics':
        return _vectorise_summary_statistics(diagram)
    elif method == 'top_persistence':
        return _vectorise_top_persistence(diagram)


def _vectorise_summary_statistics(diagram):

    statistic_fn = [diagram.total_persistence, diagram.infinity_norm]
    params = [1.0, 2.0]

    # Go over all combinations: this will first exhaust the statistic
    # functions and *then* exhaust the parameters.
    X = [fn(p) for fn, p in itertools.product(statistic_fn, params)]
    return X


def _vectorise_top_persistence(diagram):
    # TODO: make configurable
    n_features = 5
    return featurise_distances(diagram)[:n_features]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('DIRECTORY', type=str, help='Input directory')
    parser.add_argument(
        '-d',
        '--dimensions',
        type=list,
        default=[2],
        help='Indicates which dimensions to use'
    )

    parser.add_argument(
        '-m',
        '--method',
        type=str,
        default='summary_statistics',
        help='Indicates which vectorisation method to use'
    )

    args = parser.parse_args()

    # This will later become the large feature matrix (or distance
    # matrix) for clustering subjects. At the outset, it will only
    # contain features per subject (as a large vector), which will
    # subsequently be unravelled, though.
    X = collections.defaultdict(list)

    filenames = sorted(
        glob.glob(
            os.path.join(args.DIRECTORY, '*.json')
        )
    )

    for filename in tqdm(filenames, desc='File'):
        subject, _, time = parse_filename(filename)

        assert subject is not None
        assert time is not None

        persistence_diagrams = load_persistence_diagram_json(
            filename, return_raw=False
        )

        # Remove diagrams that are unnecessary
        persistence_diagrams = [
            D for D in persistence_diagrams if D.dimension in args.dimensions
        ]

        # Will contain the feature vector for each subject; the
        # underlying assumption is that all subjects have equal
        # time series lengths.
        #
        # The ordering of time points is ensured because of the
        # sorting procedure.
        x = []

        for diagram in persistence_diagrams:
            if diagram.dimension in args.dimensions:
                x.extend(vectorise_diagram(diagram, args.method))

        X[subject].extend(x)

    y = list(X.keys())                  # subject labels
    X = np.asarray(list(X.values()))    # feature vectors

    clf = AgglomerativeClustering(
        distance_threshold=0,  # compute full dendrogram
        n_clusters=None        # do not limit number of clusters
    )
    model = clf.fit(X)
    M = get_linkage_matrix(model)

    dimensions_str = '_'.join([str(d) for d in args.dimensions])
    out_filename = f'Linkage_matrix_{args.method}_d_{dimensions_str}.txt'

    np.savetxt(out_filename, M)
    np.savetxt('Labels.txt', y, fmt='%s')

    # Get cluster assignments for simple binary clustering; this is the
    # easiest clustering we can do here.
    clf = AgglomerativeClustering()
    y_pred = clf.fit_predict(X).tolist()

    assignments = {
        subject: label for (subject, label) in zip(y, y_pred)
    }

    with open(f'Assignments_{args.method}_d_{dimensions_str}.json', 'w') as f:
        json.dump(assignments, f, indent=4)