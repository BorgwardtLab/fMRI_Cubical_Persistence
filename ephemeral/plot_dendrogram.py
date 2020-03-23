#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from scipy.cluster.hierarchy import dendrogram


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', type=str, help='Input file')

    args = parser.parse_args()
    data = np.loadtxt(args.FILE)

    # Ensures that we do not overwrite anything for the subsequent
    # output.
    basename = os.path.basename(args.FILE)
    basename = os.path.splitext(basename)[0]
    basename = basename.replace('Linkage_matrix', 'Dendrogram')

    fig, ax = plt.subplots(figsize=(20, 10))
    dendrogram(data, truncate_mode=None, ax=ax)

    plt.title(basename)
    plt.savefig(basename + '.png')
