#!/usr/bin/env python
#
# Visualises a persistence diagram (or a set thereof) generated from
# DIPHA, the 'Distributed Persistent Homology Algorithm'. Currently,
# the visualisation will just result in writing stuff to `stdout`.


import argparse

import numpy as np

from topology import load_persistence_diagram_dipha
from topology import PersistenceDiagram


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', nargs='+', type=str)

    args = parser.parse_args()

    for filename in args.FILE:
        dimensions, C, D = load_persistence_diagram_dipha(filename)
        print(dimensions)
