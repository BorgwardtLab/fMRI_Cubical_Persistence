#!/usr/bin/env python
#
# Basic conversion script for NIfTI data sets. After specifying such
# a data set and an optional brain mask, converts everything into an
# array.

import argparse

import nilearn as nl
import numpy as np


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
            the image at which the mask was applied.

    Returns
    -------

        Image data along with the masked values. The dimensions of the
        input data will not be changed.
    '''

    image = nl.image.load_img(image_filename)
    mask = nl.image.load_img(mask_filename)

    from nilearn.input_data import NiftiMasker

    masker = NiftiMasker(mask)
    masked = masker.fit_transform(image)
    masked = masker.inverse_transform(masked)

    # We use the *original* mask to update the values in all voxels in
    # order to ensure that we do not touch *existing* data.
    masked.get_fdata()[np.where(mask.get_fdata() == 0)] = mask_value
    return masked


def to_dipha_format(image, filename):
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
    '''

    # TODO: replace with proper image data
    data = np.random.normal(size=(4,4))

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str)
    parser.add_argument('-m', '--mask', type=str)

    args = parser.parse_args()

    if args.mask:
        image = mask_image(args.image, args.mask)
    else:
        image = nl.image.load_img(args.image)

    from nilearn import plotting

    plotting.plot_stat_map(nl.image.index_img(image, 0))
    plotting.show()
