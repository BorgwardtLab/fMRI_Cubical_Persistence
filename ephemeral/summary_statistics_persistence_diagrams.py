#!/usr/bin/env python
#
# Calculates summary statistics of a sequence of persistence diagrams,
# calculated by DIPHA, the 'Distributed Persistent Homology Algorithm'
# library. Users can select the summary statistic that should be used.
# The following options exist:
#
#   - total persistence (with additional power arguments)
#   - the supremum norm of the diagram


import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

from topology import load_persistence_diagram_dipha
from topology import load_persistence_diagram_json
from topology import load_persistence_diagram_txt
from topology import PersistenceDiagram

from utilities import parse_filename

from tqdm import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        required=True,
        type=str,
        help='Input directory'  # FIXME: this is terse!
    )

    parser.add_argument(
        '-d', '--dimension',
        type=int,
        default=2,
        help='Dimension to select from the persistence diagrams. Only tuples '
             'of this dimension will be considered.'
    )

    parser.add_argument(
        '-s', '--statistic',
        nargs='+',
        type=str,
        default='total_persistence',
        help='Selects summary statistic to calculate for each diagram. Can '
             'be one or more of: [total_persistence, infinity_norm]'
    )

    parser.add_argument(
        '-p', '--power',
        type=float,
        default=1.0,
        help='Chooses the exponent for several summary statistics. This '
             'value might not be used for all of them.'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='Summary_statistics.json',
        help='Output file for creating summary statistics'
    )

    args = parser.parse_args()

    # Ensures that we always loop over a set of statistics, even if only
    # a single one has been specified by the client.
    if type(args.statistic) is list:
        statistics = args.statistic
    else:
        statistics = [args.statistic]

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
        else:
            load_persistence_diagram_fn = load_persistence_diagram_txt

        persistence_diagrams = load_persistence_diagram_fn(
            filename,
            return_raw=False
        )

        for diagram in persistence_diagrams:
            if diagram.dimension == args.dimension:
                diagrams_per_subject[subject].append(diagram)

    # Maps a method name to a summary statistic that should be
    # calculated.
    statistic_fn = {
        'total_persistence': PersistenceDiagram.total_persistence,
        'infinity_norm': PersistenceDiagram.infinity_norm
    }

    # Prepare keyword arguments for the summary statistics. Some of
    # these keywords might be ignored later on.
    kwargs = {
        'p': args.power
    }

    # Output for each subject, grouped according to the different
    # methods that have been selected by the user.
    data = {}

    # Calculate summary statistics for the time series of each subject
    # and print them.
    for subject in sorted(diagrams_per_subject.keys()):
        diagrams = diagrams_per_subject[subject]

        # Create a new nested hierarchy that will contain the individual
        # measurements of each summary statistic.
        data[subject] = collections.defaultdict(list)

        for diagram in diagrams:
            for statistic in statistics:
                value = statistic_fn[statistic](diagram, **kwargs)

                data[subject][statistic].append(value)

    with open(args.output, 'w') as f:
        json.dump(data, f, indent=4)
