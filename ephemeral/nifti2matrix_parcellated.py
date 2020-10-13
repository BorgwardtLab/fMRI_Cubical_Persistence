#!/usr/bin/env python
#
# Basic conversion script for NIfTI data sets, taking into account
# parcellated data and a lookup array for said data. Everything is
# converted into an array, just like `nifti2matrix.py`.

import argparse
import math
import numbers
import os
import warnings

import nilearn as nl
import numpy as np

from tqdm import tqdm

from utilities import parse_filename


def to_dipha_format(index, image, filename, parcel_map, parcel_data):
    """Convert NifTI image to DIPHA file format.

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
    index : int
        Index of participant. This is required to access the voxels
        correctly.

    image : `Nifti1Image`
        NIfTI image of arbitrary dimensions.

    filename : str
        Output filename

    parcel_map : `np.array`
        Map of voxels to parcel regions

    parcel_data : `np.array`
        Parcellated data; this is assumed to be a single array
        containing all parcels for all participants.
    """
    data = image.get_data()

    # Nothing should be overwritten. Else, the script might be used
    # incorrectly, so we refuse to do anything.
    if os.path.exists(filename):
        warnings.warn(f'File {filename} already exists. Refusing to overwrite '
                      f'it and moving on.')

        return

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
        '-p', '--parcellated',
        type=str,
        help='Parcellated values (single volume for all participants)',
        required=True
    )

    parser.add_argument(
        '-m', '--map',
        type=str,
        help='Map of voxels to parcels',
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
    image = nl.image.load_img(args.image)

    # The last dimension is always used as the time series dimension.
    # Technically, one could do additional checks here.
    n_time_steps = image.shape[-1]

    os.makedirs(args.output, exist_ok=True)

    # Required in order to obtain the proper index for accessing
    # features in the parcellated volume.
    subject, _, _ = parse_filename(args.image) 

    for index, i in enumerate(tqdm(nl.image.iter_img(image),
                                   desc='Converting', total=n_time_steps)):

        # Build a nice filename such that all output files are stored
        # correctly.
        time = format_time(index, n_time_steps)
        filename = f'{basename(args.image)}_{time}.bin'
        filename = os.path.join(args.output, filename)

        to_dipha_format(
            subject,
            i,
            f'{filename}',
            args.map,
            args.parcellated
        )
