#!/usr/bin/env python3
#
# Calculates persistence diagrams from (correlation) matrices instead of
# raw volumes. Technically, this could also be solved with DIPHA, but in
# our case, doing it directly on a graph is also possible.

import argparse

import numpy as np

from utilities import parse_filename

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', type=str, help='Input file(s)')

    args = parser.parse_args()

    for filename in args.INPUT:
        X = np.load(filename, allow_pickle=True)
        X = X['X']

        subject, _, _ = parse_filename(filename)

        # Next steps:
        # - create graph from matrix (direct conversion)
        # - calculate persistent homology (persistence diagrams)
        # - store persistence diagrams
