#!/usr/bin/env python
#
# Visualises the average persistence image of a subject sequence.

import argparse
import collections
import itertools
import glob
import os

import matplotlib.pyplot as plt

from persim import PersImage

from topology import load_persistence_diagram_json

from utilities import parse_filename

from tqdm import tqdm


def embed(subject, diagrams, suffix):
    pairs = list(itertools.chain.from_iterable(diagrams))

    clf = PersImage()
    img = clf.transform(pairs)

    path = f'../figures/persim_mean/{suffix}'

    plt.imshow(img, cmap='Spectral')

    # Create output directories for storing *all* subjects in. This
    # depends on the input file.
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f'{subject}.png'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        required=True,
        type=str,
        help='Input directory'
    )

    parser.add_argument(
        '-d', '--dimension',
        type=int,
        default=2,
        help='Dimension to select from the persistence diagrams. Only tuples '
             'of this dimension will be considered.'
    )

    parser.add_argument(
        '-s', '--suffix',
        type=str,
        help='Output suffix'
    )

    args = parser.parse_args()

    # Will store all persistence diagrams, ordered by subject. The keys
    # are the subject identifiers, extracted from the filename, whereas
    # the values are the persistence diagrams stored for them. Ordering
    # of the diagrams follows the time step information.
    diagrams_per_subject = collections.defaultdict(list)

    filenames = sorted(glob.glob(os.path.join(args.input, '*.json')))
    for filename in tqdm(filenames, desc='File'):

        subject, _, time = parse_filename(filename)
        extension = os.path.splitext(filename)[1]

        persistence_diagrams = load_persistence_diagram_json(
            filename,
            return_raw=False
        )

        for diagram in persistence_diagrams:
            if diagram.dimension == args.dimension:
                # TODO: the vectoriser could suport different
                # dimensions, but I pick a single one.
                diagrams_per_subject[subject].append(
                    diagram.toarray()
                )

    for subject in diagrams_per_subject.keys():
        embed(subject, diagrams_per_subject[subject], args.suffix)
