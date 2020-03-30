#!/usr/bin/env python
#
# Basic conversion script for NIfTI data sets. After specifying such
# a data set and an optional brain mask, converts everything into an
# array.

import argparse
import math
import numbers
import os
import warnings

import nilearn as nl
import numpy as np


from tqdm import tqdm


def mask_image(image_filename, mask_filename, mask_value=np.nan):
    '''
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

        mask_value:
            Optional value that will be used to indicate the regions in
            the image at which the mask was applied. This can be either
            a number, in which case everything is set accordingly, or a
            string (either `min` or `max`), in which case the *minimum*
            or *maximum* value of the volume is used.

    Returns
    -------

        Image data along with the masked values. The dimensions of the
        input data will not be changed.
    '''

    image = nl.image.load_img(image_filename)
    mask = nl.image.load_img(mask_filename)

    # Ensures that the mask only consists of two values. This enables the
    # conversion of malformed masks.
    mask.get_data()[np.where(mask.get_data() != 0)] = 1

    from nilearn.input_data import NiftiMasker

    masker = NiftiMasker(mask_img=mask)
    masked = masker.fit_transform(image)
    masked = masker.inverse_transform(masked)

    # No global mask value has been supplied, so we check whether it is
    # a string and proceed from there.
    if not isinstance(mask_value, numbers.Number):

        assert mask_value in ['min', 'max']

        if mask_value == 'min':
            mask_value_ = np.min(masked.get_data())
        else:
            mask_value_ = np.max(masked.get_data())

    # Set local mask value to assign. This is for convenience reasons
    # such that we do not have to duplicate the actual assignment.
    else:
        mask_value_ = mask_value

    # We use the *original* mask to update the values in all voxels in
    # order to ensure that we do not touch *existing* data.
    masked.get_data()[np.where(mask.get_data() == 0)] = mask_value_
    return masked


def to_dipha_format(image, filename, superlevel=False, normalise=False):
    '''
    Converts a NIfTI image to the DIPHA file format. The file is useful
    for describing a d-dimensional grey-scale image data. It requires a
    set of header values:

    - The 'magic number' 8067171840 (64-bit int)
    - 1 (64-bit int)
    - Total number of values (64-bit int)
    - Dimension, i.e. how many axes there are (64-bit int)
    - Set of resolutions or 'shape' values (64-bit int each)
    - The values of the array in scan-line order

    Parameters
    ----------

        image:
            NIfTI image of arbitrary dimensions.

        filename:
            Output filename

        superlevel:
            Optional flag. If set, scales all values by negative one to
            create a superlevel set filtration.

        normalise:
            Optional flag. If set, normalises the activation values of
            every voxel such that the mean activation over time, which
            is taken to be the last axis of the image, is zero and the
            variance is one.
    '''

    data = image.get_data()

    if superlevel:
        data *= -1.0

    # This assumes that the last axis is the one containing all time
    # information. Should this be made configurable?
    if normalise:
        mean = np.mean(data, axis=-1)
        variance = np.var(data, axis=-1)

        # Just to be on the safe side here...
        assert mean.shape == variance.shape

        n_invalid = np.sum(np.isnan(data))

        if n_invalid != 0:
            warnings.warn(f'File contains {n_invalid} NaN value(s) '
                          f'*prior* to normalisation.')

        # Automated broadcasting will not work because the mean has
        # a different shape. Ditto for variance.
        data -= mean[..., np.newaxis]
        data /= variance[..., np.newaxis]

        n_invalid = np.sum(np.isnan(data))

        if n_invalid != 0:
            warnings.warn(f'File contains {n_invalid} NaN value(s) '
                          f'*after* normalisation.')

    with open(filename, 'wb') as f:
        magic_number = np.int64(8067171840)
        file_id = np.int64(1)
        n_values = np.int64(len(data.ravel()))
        dimension = np.int64(len(data.shape))

        header = [magic_number, file_id, n_values, dimension]

        for h in header:
            f.write(h)

        for resolution in data.shape:
            f.write(np.int64(resolution))

        # The output stream of values has to written out as
        # double-precision floating point values. These are
        # following the pre-defined scan-line order.
        for x in data.ravel():
            f.write(np.double(x))


def basename(filename):
    '''
    Removes all extensions from a filename and returns the basename of
    the file. This function is required to handle filenames with *two*
    or more extensions.
    '''

    filename = os.path.basename(filename)

    def _split_extension(filename):
        return os.path.splitext(filename)

    filename, extension = _split_extension(filename)

    while extension:
        filename, extension = _split_extension(filename)

    return filename


def format_time(time, n_time_steps):
    '''
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
    '''

    n_digits = int(math.log10(n_time_steps) + 1)
    return f'{time:0{n_digits}d}'


if __name__ == '__main__':

    # `nilearn` and `nibabel` have a compatability issue concerning the
    # `get_data()` function. Until this bug has been rectified, we will
    # use the old behaviour and ignore all warnings.
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--image',
        type=str,
        help='NIfTI image file'
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
