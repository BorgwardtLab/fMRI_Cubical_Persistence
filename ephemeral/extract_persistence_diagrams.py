#!/usr/bin/env python3
#
# Extracts persistence diagrams from a sequence of JSON files, applying
# additional optional subsampling steps in order to obtain an improved
# visualisation.

import argparse
import json

import numpy as np

from topology import load_persistence_diagram_json
from utilities import parse_filename


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'FILES',
        nargs='+',
        type=str,
        help='Input file(s)'
    )

    args = parser.parse_args()

    for filename in sorted(args.FILES):

        _, _, t = parse_filename(filename)
        t = int(t)

        dimension, creation, destruction = \
            load_persistence_diagram_json(filename)

        creation = creation[dimension == 2]
        destruction = destruction[dimension == 2]
        time = [t] * len(creation)

        pd = np.vstack((time, creation, destruction)).T
