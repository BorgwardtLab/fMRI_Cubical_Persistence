#!/usr/bin/env python
#
# Performs variance analysis of the whole data set, based on certain
# summary statistics.
import argparse
import collections
import glob
import json
import os

import numpy as np

from topology import load_persistence_diagram_dipha
from topology import load_persistence_diagram_json
from topology import PersistenceDiagram

from utilities import dict_to_str
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
        default=['total_persistence', 'infinity_norm'],
        help='Selects summary statistic to calculate for each diagram. Can '
             'be one or more of: [total_persistence, infinity_norm]'
    )

    parser.add_argument(
        '-p', '--power',
        type=int,
        nargs='+',
        default=[1.0, 2.0],
        help='Chooses the exponent for several summary statistics. This '
             'value might not be used for all of them.'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file for creating summary statistics',
        required=True,
    )

    args = parser.parse_args()

    # Will store all persistence diagrams, ordered by subject. The keys
    # are the subject identifiers, extracted from the filename, whereas
    # the values are the persistence diagrams stored for them. Ordering
    # of the diagrams follows the time step information.
    diagrams_per_subject = collections.defaultdict(list)

    filenames = sorted(glob.glob(os.path.join(args.input, 'sub-pixar00?*00?.json')))
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
                diagrams_per_subject[subject].append(diagram)

    # Maps a method name to a summary statistic that should be
    # calculated.
    statistic_fn = {
        'total_persistence': PersistenceDiagram.total_persistence,
        'infinity_norm': PersistenceDiagram.infinity_norm
    }

    # Output for each subject, grouped according to the different
    # methods that have been selected by the user.
    data = {}

    # Will be updated with the length of the time series as soon as the
    # first subject has been fully processed.
    n_time_points = 0

    # Calculate summary statistics for the time series of each subject
    # and print them.
    for subject in sorted(diagrams_per_subject.keys()):
        diagrams = diagrams_per_subject[subject]

        # Create a new nested hierarchy that will contain the individual
        # measurements of each summary statistic.
        data[subject] = collections.defaultdict(list)

        if n_time_points == 0:
            n_time_points = len(diagrams)

        for diagram in diagrams:
            for statistic in args.statistic:
                for power in args.power:
                    # Prepare keyword arguments for the summary statistics.
                    # Some of these keywords might be ignored later on.
                    kwargs = {
                        'p': power
                    }

                    value = statistic_fn[statistic](diagram, **kwargs)

                    # Calculates a proper key value based on the parameters
                    # of the summary statistic. This ensures that different
                    # parameters are assigned a different key.
                    key = statistic + '_' + dict_to_str(kwargs)

                    data[subject][key].append(value)

    # Collection of matrices of summary statistics; will be later
    # 'squished' to provide variance analysis
    matrices = {}

    # Start *collating* data for each time point and each summary
    # statistic. We just assume that the length of all time series
    # is the same.
    for row_index, subject in enumerate(sorted(data.keys())):
        data_per_subject = data[subject]

        for statistic in sorted(data_per_subject.keys()):
            time_series = data_per_subject[statistic]

            # Make sure that we have sufficient information to
            # add to the matrices.
            assert len(time_series) == n_time_points

            if statistic not in matrices:
                n = len(data.keys())  # no. subjects
                d = n_time_points     # no. time points

                matrices[statistic] = np.zeros((n, d))

            matrices[statistic][row_index] = time_series

    # Prepare the output file. It will not only contain information
    # about the individual matrices but also the *input* parameters
    # in order to make everything reproducible.
    data = {
        k: np.var(m, axis=0).tolist() for k, m in matrices.items()
    }

    data['input'] = args.input
    data['dimension'] = args.dimension
    data['statistic'] = args.statistic
    data['power'] = args.power

    with open(args.output, 'w') as f:
        json.dump(data, f, indent=4)
