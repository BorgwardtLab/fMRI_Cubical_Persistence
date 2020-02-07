#!/usr/bin/env python
#
# Visualises a persistence diagram (or a set thereof) generated from
# DIPHA, the 'Distributed Persistent Homology Algorithm'. Currently,
# the visualisation will just result in writing stuff to `stdout`.


import argparse

import numpy as np


def load_persistence_diagram(filename):
    '''
    Loads a persistence diagram from a file. The file is assumed to be
    in DIPHA format.

    TODO: extend this!
    '''

    def _read_int64(f):
        return np.fromfile(f, dtype=np.int64, count=1)[0]

    def _fromfile(f, dtype, count, skip):
        data = np.zeros((count, ))

        for c in range(count):
            data[c] = np.fromfile(f, dtype=dtype, count=1)[0]
            f.seek(f.tell() + skip)

        return data

    with open(filename, 'rb') as f:

        magic_number = _read_int64(f)
        file_id = _read_int64(f)

        # Ensures that this is DIPHA file containing a persistence
        # diagram, and nothing something else.
        assert magic_number == 8067171840
        assert file_id == 2

        n_pairs = _read_int64(f)

        # FIXME: this does *not* follow the original MATLAB script, but
        # it produces the proper results.
        dimensions = _fromfile(
            f,
            dtype=np.int64,
            count=n_pairs,
            skip=16
        )

        # Go back whence you came!
        f.seek(0, 0)
        f.seek(32)

        creation_values = _fromfile(
            f,
            dtype=np.double,
            count=n_pairs,
            skip=16
        )

        # Go back whence you came!
        f.seek(0, 0)
        f.seek(40)

        destruction_values = _fromfile(
            f,
            dtype=np.double,
            count=n_pairs,
            skip=16
        )

    return creation_values, destruction_values


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', nargs='+', type=str)

    args = parser.parse_args()

    for filename in args.FILE:
        C, D = load_persistence_diagram(filename)
        print(C, D)
