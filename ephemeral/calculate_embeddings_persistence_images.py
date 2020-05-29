#!/usr/bin/env python
#
# Calculates embeddings of sets of persistence diagrams, based on
# their conversion to persistence images.

import argparse
import collections
import glob
import json
import os
import persim

from topology import load_persistence_diagram_dipha
from topology import load_persistence_diagram_json

from utilities import parse_filename

from tqdm import tqdm


def create_feature_vectors(diagrams_per_subject, sigma, resolution):
    """Create feature vectors of sequence of diagrams.

    This function uses the persistence image transformation to create
    feature vectors for the diagrams.

    Parameters
    ----------
    sigma : float
        Standard deviation for the persistence image calculation

    resolution : int

        Resolution in pixels for each persistence image

    Returns
    -------
    Set of persistence images.
    """
    vectoriser = persim.PersImage(
        spread=sigma,
        pixels=(resolution, resolution)
    )

    # Follows the same indexing as the diagrams; each key is a subject,
    # while each value is a list of feature vectors.
    features_per_subject = collections.defaultdict(list)

    for subject, diagrams in tqdm(diagrams_per_subject.items()):

        # While `persim` supports multiple diagrams at once, I need to
        # save them as a list here, because I do not want to serialise
        # arrays.
        for diagram in diagrams:
            X = vectoriser.transform(diagram)
            features_per_subject[subject].append(X.ravel().tolist())

    return features_per_subject


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
        '-s', '--sigma',
        type=float,
        default=1.0,
        help='Standard deviation for persistence image calculation'
    )

    parser.add_argument(
        '-r', '--resolution',
        type=int,
        default=10,
        help='Resolution (in pixels) for persistence image calculation'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file for storing embedded points',
        required=True,
    )

    args = parser.parse_args()

    if os.path.exists(args.output):
        raise RuntimeError(f'{args.output} exists; refusing to overwrite')

    # Will store all persistence diagrams, ordered by subject. The keys
    # are the subject identifiers, extracted from the filename, whereas
    # the values are the persistence diagrams stored for them. Ordering
    # of the diagrams follows the time step information.
    diagrams_per_subject = collections.defaultdict(list)

    filenames = sorted(glob.glob(os.path.join(args.input, '*.json')))
    for filename in tqdm(filenames, desc='File'):

        subject, _, time = parse_filename(filename)
        extension = os.path.splitext(filename)[1]

        if extension == '.bin':
            load_persistence_diagram_fn = load_persistence_diagram_dipha
        elif extension == '.json':
            load_persistence_diagram_fn = load_persistence_diagram_json

        persistence_diagrams = load_persistence_diagram_fn(
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

    data = create_feature_vectors(
        diagrams_per_subject,
        args.sigma,
        args.resolution
    )

    data['input'] = args.input
    data['dimension'] = args.dimension
    data['sigma'] = args.sigma
    data['resolution'] = args.resolution

    head, _ = os.path.split(args.output)

    if head:
        os.makedirs(head, exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(data, f, indent=4)
