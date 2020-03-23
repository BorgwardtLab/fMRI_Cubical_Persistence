#!/usr/bin/env python3

import argparse
import itertools
import json
import os


def compare_clusterings(A, B):
    """Compare two clusterings."""

    y_A = sorted(A.keys())
    y_B = sorted(B.keys())

    assert y_A == y_B

    y = y_A

    n_pairs = len(y) ** 2
    n_same_cluster = 0

    for y1, y2 in itertools.product(y, y):
        a1, a2 = A[y1], A[y2]
        b1, b2 = B[y1], B[y2]

        if a1 == a2 and b1 == b2:
            n_same_cluster += 1

    return n_same_cluster / n_pairs


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
        A = clusterings[os.path.splitext(os.path.basename(F))[0]]
        B = clusterings[os.path.splitext(os.path.basename(G))[0]]

        agreement = compare_clusterings(A, B)
        print(agreement)
