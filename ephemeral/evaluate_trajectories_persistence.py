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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', type=str, help='Input file(s)')

    args = parser.parse_args()

    if args.t is None:
        args.t = args.k

    for filename in args.INPUT:
        df = pd.read_csv(filename, index_col='time')
        X = df[['x', 'y']].to_numpy()

        print(X)
