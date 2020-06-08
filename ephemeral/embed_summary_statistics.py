#!/usr/bin/env python3
#
# Embeds summary statistics curves.

import argparse
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')

    parser.add_argument(
        '-c', '--children',
        action='store_true',
        help='If set, only visualises children'
    )

    parser.add_argument(
        '-s', '--statistic',
        type=str,
        help='Summary statistic to extract',
        required=True,
    )

    args = parser.parse_args()

    with open(args.INPUT) as f:
        data = json.load(f)

    df = []

    for subject in tqdm(sorted(data.keys()), desc='Subject'):

        # Check whether we are dealing with a proper subject, or an
        # information key in the data.
        try:
            _ = int(subject)
        except ValueError:
            continue

        # Store a mean representation of the persistence image of each
        # participant. This is not necessarily the smartest choice but
        # a very simple one.
        curve = data[subject][args.statistic]
        df.append(curve)

    df = pd.DataFrame(df)
    df = (df - df.min()) / (df.max() - df.min())

    df_cohorts = pd.read_csv('../data/participant_groups.csv')
    df['cohort'] = df_cohorts['cluster']
    df['cohort'] = df['cohort'].astype('str')

    X = df.select_dtypes(np.number).to_numpy()

    df_ages = pd.read_csv('../data/participant_ages.csv')
    df['age'] = df_ages['Age']
    df['cohort'] = df['cohort'].astype(int)

    if args.children:
        X = X[df['age'] < 18]
        df = df[df['age'] < 18]

        # Just to adjust the colour map later on
        df['cohort'] -= 1

    D = pairwise_distances(X, metric='l2')
    Y = MDS(
        dissimilarity='precomputed',
        max_iter=1000,
        n_init=32,
        random_state=42,
    ).fit_transform(D)

    df['x'] = Y[:, 0]
    df['y'] = Y[:, 1]

    df = df[['x', 'y', 'age', 'cohort']]

    pal = list(reversed(sns.color_palette('Set1', n_colors=6)))
    sns.scatterplot('x', 'y', data=df, hue='cohort', palette=pal)

    output = os.path.splitext(os.path.basename(args.INPUT))[0]
    output += f'_{args.statistic}'
    output += '.csv'
    output = os.path.join('../results/embeddings/', output)

    if not os.path.exists(output):
        df.to_csv(output, index=False, float_format='%.2f')

    plt.show()
