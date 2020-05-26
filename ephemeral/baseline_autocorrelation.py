#!/usr/bin/env python
#
# Baseline calculation script for NIfTI data sets. After specifying such
# a data set and an optional brain mask, converts each participant to an
# autocorrelation-based matrix representation.
#
# The goal is to summarise each participant as a time-by-time matrix. It
# can subsequently be used in additional processing and analysis tasks.

import argparse
import math
import numbers
import os
import warnings

import nilearn as nl
import numpy as np

from tqdm import tqdm


def mask_image(image_filename, mask_filename, mask_value=np.nan):
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


def format_time(time, n_time_steps):
    """Format time index based on data availability.

    Formats a time index according to a maximum number of time steps in
    in the data. This is useful in order to ensure consistent filenames
    of the *same* length.

    Parameters
    ----------
    time:
        Current time step that should be formatted

    n_time_steps:
        Maximum number of time steps

    Returns
    -------
    Formatted string with added zeroes for padding. For example, if
    there are 23 time steps in total, time step 5 will be formatted
    as  '05', thereby ensuring that a lexicographical ordering will
    be sufficient to sort the resulting files.
    """
    n_digits = int(math.log10(n_time_steps) + 1)
    return f'{time:0{n_digits}d}'


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
        help='Optional image mask to load'
    )

    parser.add_argument(
        '--mask-value',
        type=str,
        help='Value to write to the volume whenever the optional mask '
             'would apply. Can be either a number or a string. If you '
             'supply a string, it should be either `min` or `max`, to '
             'denote the minimum or maximum image value, respectively.',
        default='min'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output directory. If not set, will default to the current '
             'directory.',
        default='.'
    )

    parser.add_argument(
        '-S', '--superlevel',
        action='store_true',
        help='If set, calculates a superlevel set filtration. This will '
             'involve scaling all values by `-1`.'
    )

    parser.add_argument(
        '-n', '--normalise',
        action='store_true',
        help='If set, normalises the intensity for each voxel such that '
             'its mean activation is zero.'
    )

    args = parser.parse_args()

    if args.mask:
        image = mask_image(args.image, args.mask, mask_value=args.mask_value)
    else:
        image = nl.image.load_img(args.image)

    # The last dimension is always used as the time series dimension.
    # Technically, one could do additional checks here.
    n_time_steps = image.shape[-1]

    os.makedirs(args.output, exist_ok=True)

    for index, i in enumerate(tqdm(nl.image.iter_img(image),
                                   desc='Converting', total=n_time_steps)):

        # Build a nice filename such that all output files are stored
        # correctly.
        time = format_time(index, n_time_steps)
        filename = f'{basename(args.image)}_{time}.bin'
        filename = os.path.join(args.output, filename)

        to_dipha_format(i, f'{filename}', args.superlevel, args.normalise)
