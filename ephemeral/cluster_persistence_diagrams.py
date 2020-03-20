#!/usr/bin/env python3

import argparse
import glob
import itertools
import os

from topology import load_persistence_diagram_json
from utilities import parse_filename

from tqdm import tqdm


def vectorise_diagram(diagram, method):
    """Vectorise persistence diagram.

    Performs persistence diagram vectorisation based on a given method
    and returns a feature vector.

    Parameters
    ----------
    diagram : topology.PersistenceDiagram
        Persistence diagram to vectorise

    method : str
        Method for vectorisation. Can be either one of 'summary_statistics'
        or 'betti_curves'. This determines *how* features are stored.

    Returns
    -------
    Feature vector corresponding to the current diagram. Can be appended
    to form large-scale feature vectors, if desired.
    """

    if method == 'summary_statistics':
        return _vectorise_summary_statistics(diagram)


def _vectorise_summary_statistics(diagram):

    statistic_fn = [diagram.total_persistence, diagram.infinity_norm]
    params = [1.0, 2.0]

    # Go over all combinations: this will first exhaust the statistic
    # functions and *then* exhaust the parameters.
    X = [fn(p) for fn, p in itertools.product(statistic_fn, params)]
    return X


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('DIRECTORY', type=str, help='Input directory')
    parser.add_argument(
        '-d',
        '--dimensions',
        type=list,
        default=[2],
        help='Indicates which dimensions to use'
    )

    parser.add_argument(
        '-m',
        '--method',
        type=str,
        default='summary_statistics',
        help='Indicates which vectorisation method to use'
    )

    args = parser.parse_args()

    filenames = sorted(glob.glob(os.path.join(args.DIRECTORY, '*.json')))
    for filename in tqdm(filenames, desc='File'):
        subject, _, time = parse_filename(filename)

        assert subject is not None
        assert time is not None

        persistence_diagrams = load_persistence_diagram_json(
            filename, return_raw=False
        )

        for diagram in persistence_diagrams:
            x = []
            if diagram.dimension in args.dimensions:
                x.append(vectorise_diagram(diagram, args.method))
