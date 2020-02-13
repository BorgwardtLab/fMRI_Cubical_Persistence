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

import numpy as np

from utilities import parse_filename
from utilities import get_patient_ids_and_times

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

    args = parser.parse_args()

    with open(args.FILE) as f:
        data = json.load(f)

    subjects = sorted(data.keys())

    for subject in subjects:

        # All the single values for the given statistic. We should make
        # sure that the numbers do not vary between subjects, but it is
        # easier to just assume that for now.
        values = data[subject][args.statistic]

        if args.standardise:
            values -= np.mean(values)
            values /= np.std(values)

        print(values)
