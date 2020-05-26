#!/usr/bin/env python3
#
# Scipt to compute the curvature of the trajectories of topological activity
# of different cohorts.

import argparse
from os.path import basename
from os.path import splitext

import pandas as pd

from curvature import  arc_chord_ratio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', type=str, help="File with X,Y coordinates and"
                        "group assignment.")

    args = parser.parse_args()
    trajs = pd.read_csv(args.FILE)

    results = []
    for cohort in trajs.cohort.unique():
        vals_all = trajs.query('cohort == @cohort')
        vals_active = trajs.query('cohort == @cohort and time >= 7')

        # Just to be sure: sort
        vals_all = vals_all.sort_values(by='time')[['x', 'y']].values
        vals_active = vals_active.sort_values(by='time')[['x', 'y']].values

        # Compute curvature
        curv_all = arc_chord_ratio(vals_all)
        curv_active = arc_chord_ratio(vals_active)
        results.append({'cohort': cohort, 'curv_all': curv_all,
                        'curv_active': curv_active})

    filename = splitext(basename(args.FILE))[0]
    results = pd.DataFrame(results)
    results.to_csv(f'../results/cohort_trajectories/{filename}_curv.csv',
                   index=False)
