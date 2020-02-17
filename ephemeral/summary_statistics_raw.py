#!/usr/bin/env python3
#
# Script for collecting raw summary statistics of the data. Everything
# is collected on a per-subject basis.

import argparse
import os

import nilearn as nl
import numpy as np


def print_summary_statistics(image):
    '''
    Prints various summary statistics about a given NIfTI image to
    `stdout`. Currently, these statistics include:

    - image shapes
    - the number of time steps
    - ranges of voxel values, reported along different axes

    Parameters
    ----------

        image:
            Input image
    '''

    shape = image.get_fdata().shape
    n_time_steps = shape[-1]

    print(f'Raw shape: {shape}')
    print(f'No. time steps: {n_time_steps}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # TODO: support multiple arguments here; essentially, we could
    # process everything iteratively...
    parser.add_argument(
        '-i', '--image',
        type=str,
        help='NIfTI image file'
    )

    args = parser.parse_args()

    image = nl.image.load_img(args.image)
    basename = os.path.basename(args.image)

    print(f'Filename: {basename}')
    print_summary_statistics(image)
