#!/usr/bin/env python3

import argparse

import numpy as np

from sklearn.manifold import MDS


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', help='Input matrix')

    args = parser.parse_args()

    D = np.loadtxt(args.INPUT)

    mds = MDS(random_state=42, dissimilarity='precomputed')
    X = mds.fit_transform(D)

    print(X)
