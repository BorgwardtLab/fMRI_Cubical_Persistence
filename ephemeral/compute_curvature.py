#!/usr/bin/env python3
#
# Scipt to compute the curvature of the trajectories of topological activity
# of different cohorts.

import argparse
from os.path import basename
from os.path import splitext

import pandas as pd
import matplotlib.pyplot as plt

from curvature import  arc_chord_ratio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', type=str, help="File with X,Y coordinates and"
                        "group assignment.")

    args = parser.parse_args()
    trajs = pd.read_csv(args.FILE)

    results = []
    single_results = []

    # Debug visualization
    #fig, axes = plt.subplots(2, 3, figsize=(7, 5), sharey=True)
    #axes = axes.flatten()
    for cohort in trajs.cohort.unique():
        vals_all = trajs.query('cohort == @cohort')
        vals_active = trajs.query('cohort == @cohort and time >= 7')

        # Just to be sure: sort
        vals_all = vals_all.sort_values(by='time')[['x', 'y']].values
        vals_active = vals_active.sort_values(by='time')[['x', 'y']].values

        # Compute curvature
        curv_all, _, _ = arc_chord_ratio(vals_all, True)
        curv_active, l_active, last = arc_chord_ratio(vals_active, True)
        results.append({'cohort': cohort, 'curv_all': curv_all,
                        'curv_active': curv_active})

        #axes[cohort].plot(l_active)
        #axes[cohort].axhline(last)
        #axes[cohort].set_title(f'Indiv. distances Cohort {cohort}')

    filename = splitext(basename(args.FILE))[0]
    results = pd.DataFrame(results)
    results.to_csv(f'../results/cohort_trajectories/{filename}_curv.csv',
                   index=False)

    #plt.tight_layout()
    #plt.savefig(f'../figures/{filename}_curv.pdf')
