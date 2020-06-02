#!/usr/bin/env python
#
# Baseline calculation script for NIfTI data sets. After specifying such
# a data set and an optional brain mask, converts each participant to an
# autocorrelation-based matrix representation.
#
# The goal is to summarise each participant as a voxel-by-voxel matrix.
#
# This script is specifically built for handling parcellated data. Other
# data sources are not possible right now, as they would probably exceed
# computational resources quickly.

import argparse
import math
import os
import warnings

import numpy as np

from tqdm import tqdm


def basename(filename):
    """Calculate basename of a file.

    Removes all extensions from a filename and returns the basename of
    the file. This function is required to handle filenames with *two*
    or more extensions.
    """
    filename = os.path.basename(filename)

    def _split_extension(filename):
        return os.path.splitext(filename)

    filename, extension = _split_extension(filename)

    while extension:
        filename, extension = _split_extension(filename)

    return filename


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output directory. If not set, will default to the current '
             'directory.',
        default='.'
    )

    args = parser.parse_args()

    data = np.load(
        '../results/parcellated_data/shaefer_masked_subject_data_shifted.npy'
    )

    n_participants = data.shape[0] + 1

    # Used to generate nice output files that follow the naming
    # convention in the remainder of the paper.
    n_digits = int(math.log10(n_participants) + 1)

    for index, X in tqdm(enumerate(data), desc='Subject'):

        filename = f'{index+1:0{n_digits}d}.npz'
        filename = os.path.join(args.output, filename)

        X = np.corrcoef(X.T)
        X = np.nan_to_num(X)

        assert np.isnan(X).sum() == 0

        # Nothing should be overwritten. Else, the script might be used
        # incorrectly, so we refuse to do anything.
        if os.path.exists(filename):
            warnings.warn(f'File {filename} already exists. Refusing to '
                          f'overwrite it and moving on.')

            continue

        np.savez(filename, X=X)
