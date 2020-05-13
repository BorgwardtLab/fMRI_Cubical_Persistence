#!/usr/bin/env python
#
# Visualises a sequence of persistence images that have been calculated
# before calling the script.

import argparse
import json
import os

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')

    # TODO: make into keyword
    show_title = False

    args = parser.parse_args()

    basename = os.path.basename(args.INPUT)
    basename = os.path.splitext(basename)[0]

    with open(args.INPUT) as f:
        data = json.load(f)

    for subject in tqdm(sorted(data.keys()), desc='Subject'):

        # Check whether we are dealing with a proper subject, or an
        # information key in the data.
        try:
            _ = int(subject)
        except ValueError:
            continue

        images = [np.array(X) for X in data[subject]]
        X = sum(images) / len(images)
        r = int(np.sqrt(X.shape[0]))
        X = X.reshape(r, r)

        output = f'../results/average_persistence_images/{basename}'
        os.makedirs(output, exist_ok=True)

        for index, image in enumerate(images):
            plt.imshow(image.reshape(r, r))
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(
                f'{output}/{subject}_{index:03d}.png',
                bbox_inches=0,
            )

        plt.imshow(X)

        if show_title:
            plt.title(subject)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{output}/{subject}.png', bbox_inches='tight')
