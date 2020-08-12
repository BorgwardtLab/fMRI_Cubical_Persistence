#!/usr/bin/env python3
#
# Creates a feature matrix from a set of persistence images.

import argparse
import json
import os

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE')

    args = parser.parse_args()

    with open(args.FILE) as f:
        data = json.load(f)

    X = []

    for key in sorted(data.keys()):
        try:
            subject = int(key)
        except ValueError:
            continue

        M = np.asarray(data[key])
        X.append(M)

    X = np.asarray(X)

    basename = os.path.basename(args.FILE)
    basename = os.path.splitext(basename)[0]
    basename = f'{basename}_feature_matrix.npz'

    if not os.path.exists(basename):
        np.savez(basename, X=X)
