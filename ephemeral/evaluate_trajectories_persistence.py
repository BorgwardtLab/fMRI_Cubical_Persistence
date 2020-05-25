#!/usr/bin/env python3
#
# Simple persistence-based analysis of trajectories. This script
# requires a working installation of Aleph. More precisely, the
# tool `vietoris_rips` must be available in the path.

import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from subprocess import check_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', type=str, help='Input file(s)')

    args = parser.parse_args()

    for filename in args.INPUT:
        df = pd.read_csv(filename, index_col='time')
        X = df[['x', 'y']].to_numpy()

        min_coordinate = np.min(np.min(X, axis=0))
        max_coordinate = np.max(np.max(X, axis=0))

        X = (X - min_coordinate) / (max_coordinate - min_coordinate)

        np.savetxt('/tmp/foo.txt', X, fmt='%.8f')

        output = check_output(
                ['vietoris_rips', '/tmp/foo.txt', '0.2', '2'],
                universal_newlines=True,
        )

        #print(X)
        print(output)

        raise 'heck'
