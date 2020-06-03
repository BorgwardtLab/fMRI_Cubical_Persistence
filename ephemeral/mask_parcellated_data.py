import numpy as np


occipital_ids = [
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0, 12.0, 14.0,
    15.0, 16.0, 19.0, 21.0, 22.0, 24.0, 28.0, 29.0, 30.0, 31.0, 35.0,
    36.0, 37.0, 39.0, 41.0, 42.0, 43.0, 48.0, 49.0, 50.0, 51.0, 52.0,
    53.0, 54.0, 55.0, 56.0, 60.0, 61.0, 62.0, 65.0, 66.0, 67.0, 69.0,
    71.0, 72.0, 73.0, 78.0, 79.0, 83.0, 87.0, 88.0, 89.0, 91.0, 96.0,
    97.0, 98.0, 99.0, 100.0
]


data = np.load(
    '../results/parcellated_data/shaefer_masked_subject_data_shifted.npy'
)

occipital_ids = np.asarray(occipital_ids, dtype=int)

occipital_mask = data.copy()
xor_mask = data.copy()

occipital_mask[:, :, ~occipital_ids] = np.nan
xor_mask[:, :, occipital_ids] = np.nan

basename = '../results/parcellated_data/shaefer_masked_subject_data_shifted'

np.save(
    basename + '_occipitalmask.npy',
    occipital_mask
)

np.save(
    basename + '_xormask.npy',
    xor_mask
)
