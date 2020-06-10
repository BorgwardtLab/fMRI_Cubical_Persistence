#!/usr/bin/env python
#
# Calculates a persistence image from a persistence diagram. This is an
# auxiliary script for creating additional figures.

import argparse
import json
import os

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

from persim import PersImage
from topology import load_persistence_diagram_txt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')
    parser.add_argument('RESOLUTION', type=int, help='Resolution')

    args = parser.parse_args()

    pd = load_persistence_diagram_txt(args.INPUT)

    clf = PersImage()
    X = clf.transform([pd])[0]

    with open(f'PI_{args.resolution}.txt', 'w') as f:
        for j in range(n):
            for i in range(n):
                print(f'{i} {j} {mean[j, i]}', file=f)
            print('', file=f)


    raise 'heck'

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

        #for index, image in enumerate(images):
        #    plt.imshow(image.reshape(r, r), cmap='Spectral')
        #    plt.axis('off')
        #    plt.tight_layout()
        #    plt.savefig(
        #        f'{output}/{subject}_{index:03d}.png',
        #        bbox_inches=0,
        #    )

        plt.imshow(X, cmap='Spectral')

        if show_title:
            plt.title(subject)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{output}/{subject}.png', bbox_inches='tight')
