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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('IMAGE')
    parser.add_argument('MASK')

    args = parser.parse_args()
    data = mask_image(args.IMAGE, args.MASK)

    from nilearn import plotting

    plotting.plot_stat_map(nl.image.index_img(data, 0))
    plotting.show()
