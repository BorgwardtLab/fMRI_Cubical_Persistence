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
from topology import PersistenceDiagram

from utilities import parse_filename

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


def make_summary_statistics_curve(diagrams, statistic_fn):
    """Create summary statistics curve.

    Parameters
    ----------
    diagrams : list of topology.PersistenceDiagram
        List of persistence diagrams to convert

    statistic_fn : callable
        Statistic to evaluate for the conversion of the list of
        persistence diagrams into a curve.

    Returns
    -------
    Curve of summary statistics.
    """
    return np.array([statistic_fn(diagram) for diagram in diagrams])


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
        default='total_persistence',
        help='Indicates which distance calculation method to use'
    )

    parser.add_argument(
        '-p',
        '--power',
        type=float,
        default='1.0',
        help='Exponent to use for calculations'
    )

    args = parser.parse_args()

    # Collects diagrams per subject, for future summary calculations.
    # This is easier than plugging together large feature vectors.
    diagrams_per_subject = collections.defaultdict(list)

    filenames = sorted(
        glob.glob(
            os.path.join(args.DIRECTORY, '*pixar00?*_00?.json')
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

        for diagram in persistence_diagrams:
            diagrams_per_subject[subject].append(diagram)

    statistic_functions = {
        'total_persistence': PersistenceDiagram.total_persistence,
        'infinity_norm': PersistenceDiagram.infinity_norm
    }

    if args.method in statistic_functions:
        curves = [
            make_summary_statistics_curve(
                diagrams_per_subject[s],
                statistic_functions[args.method]
            ) for s in sorted(diagrams_per_subject.keys())
        ]

        # TODO: make configurable and use maybe something like DTW in
        # order to capture differences between clusterings better?
        D = pairwise_distances(curves)

    # Use subject labels as 'true' labels (even though we have no way of
    # telling in a clustering setup)
    y = list(diagrams_per_subject.keys())

    clf = AgglomerativeClustering(
        distance_threshold=0,      # compute full dendrogram
        n_clusters=None,           # do not limit number of clusters
        affinity='precomputed',    # use our distances from above
        linkage='average',         # cannot use Ward linkage here
    )
    model = clf.fit(D)
    M = get_linkage_matrix(model)

    dimensions_str = '_'.join([str(d) for d in args.dimensions])
    out_filename = f'Linkage_matrix_representation_'\
                   f'{args.method}_p{args.power}_d_{dimensions_str}.txt'

    np.savetxt(out_filename, M)
    np.savetxt('Labels.txt', y, fmt='%s')

    # Get cluster assignments for simple binary clustering; this is the
    # easiest clustering we can do here.
    clf = AgglomerativeClustering(affinity='precomputed', linkage='average')
    y_pred = clf.fit_predict(D).tolist()

    assignments = {
        subject: label for (subject, label) in zip(y, y_pred)
    }

    output_filename = f'Assignments_representation_'\
                      f'{args.method}_p{args.power}_d_{dimensions_str}.json'

    with open(output_filename, 'w') as f:
        json.dump(assignments, f, indent=4)