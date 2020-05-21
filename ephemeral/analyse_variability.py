#!/usr/bin/env python3
#
# Analyses the variability of representations (either summary statistic
# curves or persistence images) for the data set.

import argparse
import collections
import glob
import json
import os

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

    # First dimension of `X` represents the time steps. Need to
    # calculate the mean over the remaining dimensions.

    print(X.shape)
