#!/usr/bin/env python3

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scipy.cluster.hierarchy import set_link_color_palette
from scipy.cluster.hierarchy import dendrogram


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', nargs='+', type=str, help='Input file')

    args = parser.parse_args()

    for filename in args.FILE:
        data = np.loadtxt(filename)

        # Ensures that we do not overwrite anything for the subsequent
        # output.
        basename = os.path.basename(filename)
        basename = os.path.splitext(basename)[0]
        basename = basename.replace('Linkage_matrix', 'Dendrogram')

        fig, ax = plt.subplots(figsize=(20, 10))
        dendrogram(
            data,
            truncate_mode=None,
            ax=ax,
            show_leaf_counts=True,
        )

        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.get_yaxis().set_ticks([])

        plt.title(basename)
        plt.tight_layout()
        plt.savefig(basename + '.png', bbox_inches='tight')
