#!/usr/bin/env python
#
# Calculates embeddings of sets of persistence diagrams, based on
# a pre-defined feature creation strategy.

import argparse
import collections
import glob
import json
import os
import pervect

from topology import load_persistence_diagram_dipha
from topology import load_persistence_diagram_json

from utilities import parse_filename

from tqdm import tqdm


def create_feature_vectors(diagrams_per_subject):
    """Create feature vectors of sequence of diagrams."""
    vectoriser = pervect.PersistenceVectorizer(
            n_components=20,  # default settings, but just to be sure
            random_state=42
    )

    # Follows the same indexing as the diagrams; each key is a subject,
    # while each value is a matrix containing features in its columns
    # and time steps in its rows.
    features_per_subject = {}

    for subject, diagrams in diagrams_per_subject.items():

        X = vectoriser.fit_transform(diagrams)
        features_per_subject[subject] = X.tolist()

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
        '-o', '--output',
        type=str,
        help='Output file for storing embedded points',
        required=True,
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

    data = create_feature_vectors(diagrams_per_subject)
    data['input'] = args.input
    data['dimension'] = args.dimension

    with open(args.output, 'w') as f:
        json.dump(data, f, indent=4)
