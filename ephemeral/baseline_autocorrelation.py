#!/usr/bin/env python
#
# Baseline calculation script for NIfTI data sets. After specifying such
# a data set and an optional brain mask, converts each participant to an
# autocorrelation-based matrix representation.
#
# The goal is to summarise each participant as a time-by-time matrix. It
# can subsequently be used in additional processing and analysis tasks.

import argparse
import os
import warnings
import sys

import nilearn as nl
import numpy as np


def mask_image(image_filename, mask_filename):
    """Mask image and return it.

    Given an image filename and a mask filename, performs the masking
    operation with a pre-defined value and returns the resulting data
    array. The function is fully agnostic with respect to dimensions;
    any image file that can be processed by `nilearn` can be used.

    Parameters
    ----------
    image_filename:
        Filename for image data

    mask_filename:
        Filename for mask data

    Returns
    -------
    Image data along with the masked values. The dimensions of the
    input data will not be changed, but reshaping is permitted, as
    this makes it easier to calculate correlations.
    """
    image = nl.image.load_img(image_filename)
    mask = nl.image.load_img(mask_filename)

    # Ensures that the mask only consists of two values. This enables the
    # conversion of malformed masks.
    mask.get_data()[np.where(mask.get_data() != 0)] = 1

    from nilearn.input_data import NiftiMasker

    masker = NiftiMasker(mask_img=mask)
    masked = masker.fit_transform(image)

    return masked


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

    # `nilearn` and `nibabel` have a compatibility issue concerning the
    # `get_data()` function. Until this bug has been rectified, we will
    # use the old behaviour and ignore all warnings.
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--image',
        type=str,
        help='NIfTI image file',
        required=True
    )

    parser.add_argument(
        '-m', '--mask',
        type=str,
        help='Image mask to load',
        required=True
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output directory. If not set, will default to the current '
             'directory.',
        default='.'
    )

    args = parser.parse_args()

    # Build a nice filename such that all output files are stored
    # correctly.
    filename = f'{basename(args.image)}.npz'
    filename = os.path.join(args.output, filename)

    # Nothing should be overwritten. Else, the script might be used
    # incorrectly, so we refuse to do anything.
    if os.path.exists(filename):
        warnings.warn(f'File {filename} already exists. Refusing to overwrite '
                      f'it and moving on.')

        sys.exit(-1)

    image = mask_image(args.image, args.mask)
    X = np.corrcoef(image)

    np.savez(filename, X=X)
