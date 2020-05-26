#!/usr/bin/env python3
#
# Analyses the variability of representations (either summary statistic
# curves or persistence images) *across* cohorts for the data set.

import argparse
import json

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')

    parser.add_argument(
        '-s', '--statistic',
        help='Summary statistic to use. If not set, defaults to assuming a '
             'persistence image data structure.',
        type=str,
        default='',
    )

    parser.add_argument(
        '-r', '--rolling',
        help='If set, uses a rolling window to smooth data.',
        type=int,
        default=0
    )

    # TODO: this should be made configurable
    parser.add_argument(
        '-d', '--drop',
        action='store_true',
        help='If set, drops measurements unrelated to the movie'
    )

    args = parser.parse_args()

    with open(args.INPUT) as f:
        data = json.load(f)

    X = []

    for subject in tqdm(sorted(data.keys()), desc='Subject'):

        # Check whether we are dealing with a proper subject, or an
        # information key in the data.
        try:
            _ = int(subject)
        except ValueError:
            continue

        # Deal with a persistence image structure, direct extraction is
        # possible.
        if not args.statistic:

            # Remove data from the relevant statistic if so desired by
            # the client. This makes sense if not all time steps are
            # relevant for the experiment.
            #
            # TODO: make extent of drop configurable
            if args.drop:
                x = data[subject][7:]
            else:
                x = data[subject]

            if args.rolling == 0:
                X.append(x)
            else:
                df = pd.DataFrame(x)
                df = df.rolling(args.rolling, axis=0, min_periods=1).mean()

                X.append(df.to_numpy())

        # Deals with a summary statistic, requires a proper selection
        # first.
        else:

            # Remove data from the relevant statistic if so desired by
            # the client. This makes sense if not all time steps are
            # relevant for the experiment.
            #
            # TODO: make extent of drop configurable
            if args.drop:
                x = data[subject][args.statistic][7:]
            else:
                x = data[subject][args.statistic]

            if args.rolling == 0:
                X.append(x)
            else:
                df = pd.Series(x)
                df = df.rolling(args.rolling, axis=0, min_periods=1).mean()

                X.append(df.to_numpy())

    # This is a tensor of shape (n, m, f) or (n, m), where $n$ represents
    # the number of subjects, $m$ the number of time steps, and $f$ the
    # number of features. In case summary statistics are being used, this
    # tensor will only be 2D because there is only a single feature.
    X = np.array(X)
    df_groups = pd.read_csv('../data/participant_groups.csv')
    cohorts = df_groups['cluster'].values

    assert len(cohorts) == X.shape[0]

    for cohort in sorted(set(cohorts)):
        cohort_mean = np.mean(X[cohorts == cohort], axis=0)

        # Calculate distance for each individual and each time step,
        # using the Euclidean distance between feature descriptors.
        distances = np.sqrt(np.sum(
            np.abs(X[cohorts == cohort] - cohort_mean)**2, axis=-1
        ))

        # `distances` now has the shape of $(n, m)$, according to the
        # definitions above. We make them comparable *across* cohorts
        # by normalising them between (0, 1).
        #
        # This can be done for every time step because each time step
        # can be considered independently.
        #
        # This amounts to a *sorting* of subjects.
        distances = ((distances - distances.min(axis=0))
                     / (distances.max(axis=0) - distances.min(axis=0)))

        for i in range(distances.shape[0]):
            plt.plot(distances[i])

        plt.show()

    raise 'heck'

    # First dimension of `X` represents the subject, which is the axis
    # we to calculate the mean over in all cases.
    mu = np.mean(X, axis=0)

    D = []

    # Report distance to the mean for each subject
    for index, row in enumerate(X):
        if not args.statistic:
            distances = np.sqrt(np.sum(np.abs(row - mu)**2, axis=-1))
        else:
            distances = np.abs(row - mu)

        D.append(distances)

    D = np.array(D)

    df = pd.DataFrame(D)

    
    df['cohort'] = df_groups['cluster']
    df['cohort'] = df['cohort'].transform(lambda x: 'g' + str(x))

    def _normalise_cohort(df):
        df = df.select_dtypes(np.number)
        df = (df - df.min()) / (df.max() - df.min())
        return df

    # Make the cohort curves configurable; since we are only showing
    # relative variabilities, this is justified.
    df.loc[:, df.columns.drop('cohort')] = df.groupby('cohort').apply(
        _normalise_cohort
    )

    df = df.groupby('cohort').agg(np.std)                  \
        .apply(lambda x: (x - x.mean()) / x.std(), axis=1) \
        .reset_index().melt(
            'cohort',
            var_name='time',
            value_name='std'
    )

    if args.drop:
        df['time'] += 7

    g = sns.FacetGrid(df, col='cohort', height=2, aspect=3)
    g.map(sns.lineplot, 'time', 'std')

    plt.tight_layout()
    plt.show()
