#!/usr/bin/env python3
#
# Analyses the variability of representations (either summary statistic
# curves or persistence images) for the data set.

import argparse
import collections
import glob
import json
import os

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')

    parser.add_argument(
        '-s', '--statistic',
        help='Summary statistic to use. If not set, defaults to assuming a '
             'persistence image data structure.',
        type=str,
        default='',
    )

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

        # Deal with a persistence image structure, direct extraction is
        # possible.
        if not args.statistic:
            X.append(data[subject])

        # Deals with a summary statistic, requires a proper selection
        # first.
        else:
            X.append(data[subject][args.statistic])

    X = np.array(X)

    # First dimension of `X` represents the subject, which is the axis
    # we to calculate the mean over in all cases.
    mu = np.mean(X, axis=0)

    D = []

    # Report distance to the mean for each subject
    for index, row in enumerate(X):
        if not args.statistic:
            distances = np.sqrt(np.sum(np.abs(row - mu)**2, axis=-1))
        else:
            distances = np.abs(row - mu)

        D.append(distances)

    D = np.array(D)
    D = (D - np.min(D, axis=0)) / (np.max(D, axis=0) - np.min(D, axis=0))

    df = pd.DataFrame(D)
    df.std().plot()

    plt.show()
