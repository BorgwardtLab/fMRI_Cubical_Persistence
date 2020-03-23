#!/usr/bin/env python3

import argparse
import itertools
import json
import os

from sklearn.metrics import adjusted_rand_score


def compare_clusterings(A, B):
    """Compare two clusterings."""
    y_A = sorted(A.keys())
    y_B = sorted(B.keys())

    assert y_A == y_B

    return adjusted_rand_score(
        list(A.values()),
        list(B.values())
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('FILES', nargs='+', type=str, help='Input files(s)')

    args = parser.parse_args()

    # Will contain clusterings for each filename. Stores the assignments
    # of subjects in a nested dictionary.
    clusterings = {}

    for filename in args.FILES:
        with open(filename) as f:
            data = json.load(f)

        basename = os.path.basename(filename)
        basename = os.path.splitext(filename)[0]

        clusterings[basename] = data

    for F, G in itertools.combinations(args.FILES, 2):

        f = os.path.splitext(os.path.basename(F))[0]
        g = os.path.splitext(os.path.basename(G))[0]

        A = clusterings[f]
        B = clusterings[g]

        f = f.replace('Assignments_representation_', '')
        f = f.replace('Assignments_', '')
        g = g.replace('Assignments_representation_', '')
        g = g.replace('Assignments_', '')

        agreement = compare_clusterings(A, B)
        print(f'{f} vs. {g}: {agreement:.2f}')
