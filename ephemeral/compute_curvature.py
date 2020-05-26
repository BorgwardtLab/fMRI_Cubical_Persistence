#!/usr/bin/env python3
#
# Scipt to compute the curvature of the trajectories of topological activity
# of different cohorts.

import argparse
from os.path import basename
from os.path import splitext

import pandas as pd
import matplotlib.pyplot as plt
from IPython import embed

from curvature import  arc_chord_ratio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', type=str, help="File with X,Y coordinates and"
                        "group assignment.")
    parser.add_argument('-w', '--window', type=int, help="Time window length"
                       " over which the curvatue is computed. If 0, curvature"
                        " is computed over the whole TS.")

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
        if args.window:
            # First for all values
            end = vals_all.shape[0] - args.window
            for idx, start in enumerate(range(end + 1)):
                v_all = vals_all[start:start + args.window]
                curv_all = arc_chord_ratio(v_all)
                results.append({'cohort': cohort, 'idx': idx,
                                  'curv_all': curv_all})

            # Then for active values
            end = vals_active.shape[0] - args.window
            for idx, start in enumerate(range(end + 1)):
                v_active = vals_active[start:start + args.window]
                curv_active = arc_chord_ratio(v_active)
                results.append({'cohort': cohort, 'idx': idx,
                                  'curv_active': curv_active})

        else:
            curv_all, _, _ = arc_chord_ratio(vals_all, True)
            curv_active, l_active, last = arc_chord_ratio(vals_active, True)
            results.append({'cohort': cohort, 'curv_all': curv_all,
                            'curv_active': curv_active})

        #axes[cohort].plot(l_active)
        #axes[cohort].axhline(last)
        #axes[cohort].set_title(f'Indiv. distances Cohort {cohort}')

    filename = splitext(basename(args.FILE))[0]
    filename += f'_{args.window}' if args.window else ''
    results = pd.DataFrame(results)
    results.to_csv(f'../results/cohort_trajectories/{filename}_curv.csv',
                   index=False)

    #plt.tight_layout()
    #plt.savefig(f'../figures/{filename}_curv.pdf')
