#!/usr/bin/env python3
#
# Calculates the XOR of two directories of image masks.

import glob
import os

import nibabel as nb
import nilearn as nl
import numpy as np

from utilities import parse_filename


if __name__ == '__main__':

    dir_a = '../data/brainmask'
    dir_b = '../data/occipitalmask'

    files_a = sorted(glob.glob(os.path.join(dir_a, 'sub*.gz')))
    files_b = sorted(glob.glob(os.path.join(dir_b, 'sub*.gz')))

    for file_a, file_b in zip(files_a, files_b):
        a, _, _ = parse_filename(file_a)
        b, _, _ = parse_filename(file_b)

        assert a == b

        mask_a = nl.image.load_img(file_a)
        mask_b = nl.image.load_img(file_b)

        # Ensures that the mask only consists of two values. This enables the
        # conversion of malformed masks.
        for mask in [mask_a, mask_b]:
            mask.get_data()[np.where(mask.get_data() != 0)] = 1

        data = (mask_a.get_data() != mask_b.get_data()).astype(np.float)
        xor_mask = nb.Nifti1Image(data, mask_a.affine, mask_a.header)

        filename = f'sub-pixar{a}_task-pixar_bold_space-MNI152NLin2009cAsym'
        filename += '_xormask.nii.gz'
        xor_mask.to_filename(filename)
