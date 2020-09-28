#!/usr/bin/env python
#
# Basic conversion script for parcellated matrices that are *not* in
# NIfTI format but need to converted to DIPHA.

import argparse
import os
import warnings

import numpy as np

from tqdm import tqdm


def to_dipha_format(matrix, directory):
    """Convert matrix to DIPHA file format.

    Converts a time-varying matrix to the DIPHA file format. This format
    is used in describing a d-dimensional grey-scale image data. We need
    set of header values:

    - The 'magic number' 8067171840 (64-bit int)
    - 1 (64-bit int)
    - Total number of values (64-bit int)
    - Dimension, i.e. how many axes there are (64-bit int)
    - Set of resolutions or 'shape' values (64-bit int each)
    - The values of the array in scan-line order

    Parameters
    ----------
    matrix:
        Input matrix of shape (N, T, M), where N is the number of
        participants of a study, T is the number of time steps, and M is
        the number of voxels. M must be a square number such that it is
        possible to reshape the matrix into a proper rectangular array.

    directory:
        Output directory
    """
    os.makedirs(directory, exist_ok=True)

    assert len(matrix.shape) == 3

    N = matrix.shape[0]
    T = matrix.shape[1]
    M = matrix.shape[2]

    # Poor man's version of checking that the number is indeed a square
    # number.
    r = np.sqrt(M)
    assert M == int(r + 0.5) ** 2

    r = int(r)

    # Participants
    for n in tqdm(range(N), desc='Particpant'):
        # Time steps
        for t in range(T):
            # TODO: this is somewhat controversial because it does *not*
            # take any proper connections into account.
            volume = matrix[n, t, :].reshape(r, r)

            # FIXME: this filename is restricted to the 'Partly Cloudy'
            # task; since this script is only meant as an *additional*
            # baseline, we can always make it more flexible later on.
            filename = f'sub-pixar{n:03d}_task-parcellated_{t:03d}.bin'

            filename = os.path.join(
                directory,
                filename
            )

            # Nothing should be overwritten. Else, the script might be used
            # incorrectly, so we refuse to do anything.
            if os.path.exists(filename):
                warnings.warn(f'File {filename} already exists. Refusing to '
                              f'overwrite it and moving on.')

                return

            with open(filename, 'wb') as f:
                magic_number = np.int64(8067171840)
                file_id = np.int64(1)
                n_values = np.int64(len(volume.ravel()))
                dimension = np.int64(len(volume.shape))

                header = [magic_number, file_id, n_values, dimension]

                for h in header:
                    f.write(h)

                for resolution in volume.shape:
                    f.write(np.int64(resolution))

                # The output stream of values has to written out as
                # double-precision floating point values. These are
                # following the pre-defined scan-line order.
                for x in volume.ravel():
                    f.write(np.double(x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', type=str, help='Input file')
    parser.add_argument('OUTDIR', type=str, help='Output directory')

    args = parser.parse_args()

    matrix = np.load(args.FILE)
    to_dipha_format(matrix, args.OUTDIR)
