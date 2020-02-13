#!/usr/bin/env python
#
# Visualises a set of summary statistics of persistence diagrams. Loads
# an input file in JSON format and reports a user-defined measure. This
# is achieved by plotting the resulting time series for all subjects in
# the data set.

import argparse
import collections
import glob
import json
import os
import sys

import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', type=str, help='Input file')

    parser.add_argument(
        '-S-', '--standardise',
        action='store_true',
        help='If set, standardises measurements per subject before reporting '
             'them in the output'
    )

    parser.add_argument(
        '-s', '--statistic',
        default='total_persistence',
        type=str,
        help='Selects summary statistic to calculate for each diagram. Must '
             'be a key that occurs in all subjects of the input file.'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file'
    )

    args = parser.parse_args()

    with open(args.FILE) as f:
        data = json.load(f)

    subjects = sorted(data.keys())

    title = args.statistic
    if args.standardise:
        title += ' (standardise)'

    fig = plt.figure(figsize=(120, 40))
    fig.suptitle(title)

    for index, subject in enumerate(subjects):

        # All the single values for the given statistic. We should make
        # sure that the numbers do not vary between subjects, but it is
        # easier to just assume that for now.
        values = data[subject][args.statistic]

        if args.standardise:
            values -= np.mean(values)
            values /= np.std(values)

        ax = fig.add_subplot(6, 5, index + 1)
        ax.set_title(f'Subject: {subject}')

        ax.plot(values)

    plt.tight_layout(h_pad=5)
    plt.savefig(args.output)
