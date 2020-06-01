#!/usr/bin/env python3
#
# Performs peri-event histogram analysis for a variability curve, based
# on bootstrapping statistics.
#
# This is based on code that was kindly contributed by Tristan Yates.

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

from tqdm import tqdm


def get_salience_indices(threshold=7):
    """Return indices with salient events (using an optional threshold).

    Parameters
    ----------
    threshold : int
        Threshold for salience calculation. If the number of votes is
        greater than or equal to the threshold, the event is reported
        in the result.

    Returns
    -------
    Indices at which salient events happen.
    """
    df_annot = pd.read_excel('../data/annotations.xlsx')

    # Get the salience values; it is perfectly justified to replace NaN
    # values by zero because those salience values will not be counted.
    salience = df_annot['Boundary salience (# subs out of 22)'].values
    salience = np.nan_to_num(salience)

    # These are the detected event boundaries, according to Tristan's
    # analysis. Note that since the index has been shifted above, the
    # dropping operation does *not* have to be considered here!
    salience_indices, = np.nonzero(salience >= threshold)

    # Again, no shift here!
    return salience_indices


def get_variability_correlation(events, curve, w):
    """Calculate correlation of pre-event and post-event variability.

    The basic idea behind this function is to take a set of events, or
    rather event boundaries, and evaluate the variability of the curve
    in terms of its mean across a window pre-event and post-event. The
    resulting numbers will be used to fit a linear model. To evaluate,
    or gauge the variability, the correlation coefficient is returned.

    This function can easily be used in a bootstrap setting by using a
    random selection of events.

    Parameters
    ----------
    events : `numpy.array`
        Sequence of indices for event boundaries

    curve : `pandas.DataFrame`
        Variability curve; must measure some scalar attribute so that
        the mean can be calculated over all events.

    w : int
        Window width; setting this to `w` means that `2*w + 1` many
        points in the curve will be evaluated.

    Returns
    -------
    Coefficient of determination for the specified data.
    """
    # This is the maximum time stored in the data set; it does not
    # necessarily correspond to the length of the curve because it
    # is possible that indices have been dropped.
    #
    # We need to do the same thing for the minimum time.
    max_t = curve['time'].max()
    min_t = curve['time'].min()

    # Makes it easier to access a given time step; note that we are
    # using the indices from the event boundaries here.
    curve = curve.set_index('time')

    # Collect peri-event statistics
    peri_event = np.zeros((w*2 + 1, len(events)))

    for idx, t in enumerate(range(-w, w+1)):
        for eb, bound in enumerate(events):
            if bound + t < min_t or bound + t > max_t:
                peri_event[idx, eb] = np.nan
            else:
                peri_event[idx, eb] = curve.loc[bound + t]

    X = np.asarray(np.arange(2*w + 1)).reshape(-1, 1)
    y = np.nanmean(peri_event, axis=1).reshape(-1, 1)

    clf = LinearRegression(normalize=True)
    clf.fit(X, y)

    return clf.score(X, y)


def get_variability_mean_difference(events, curve, w):
    """Calculate mean difference in pre-event and post-event variability.

    The basic idea behind this function is to take a set of events, or
    rather event boundaries, and evaluate the variability of the curve
    in terms of its mean across a window pre-event and post-event. The
    difference of the resulting numbers, or rather their mean, will be
    used as a test statistic and returned.

    This function can easily be used in a bootstrap setting by using a
    random selection of events.

    Parameters
    ----------
    events : `numpy.array`
        Sequence of indices for event boundaries

    curve : `pandas.DataFrame`
        Variability curve; must measure some scalar attribute so that
        the mean can be calculated over all events.

    w : int
        Window width; setting this to `w` means that `2*w + 1` many
        points in the curve will be evaluated.

    Returns
    -------
    Difference between pre-event mean and post-event mean.
    """
    # This is the maximum time stored in the data set; it does not
    # necessarily correspond to the length of the curve because it
    # is possible that indices have been dropped.
    #
    # We need to do the same thing for the minimum time.
    max_t = curve['time'].max()
    min_t = curve['time'].min()

    # Makes it easier to access a given time step; note that we are
    # using the indices from the event boundaries here.
    curve = curve.set_index('time')

    # Collect peri-event statistics
    peri_event = np.zeros((w*2 + 1, len(events)))

    for idx, t in enumerate(range(-w, w+1)):
        for eb, bound in enumerate(events):
            if bound + t < min_t or bound + t > max_t:
                peri_event[idx, eb] = np.nan
            else:
                peri_event[idx, eb] = curve.loc[bound + t]

    means = np.nanmean(peri_event, axis=1)

    # Calculate pre-event mean and post-event mean. Notice that we are
    # not calculating the mean variability at the event point itself.
    pre_event_mean = np.mean(means[:w])
    post_event_mean = np.mean(means[w+1:])

    return pre_event_mean - post_event_mean


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')
    parser.add_argument(
        '-w', '--window',
        default=3,
        type=int,
        help='Specifies window width'
    )

    parser.add_argument(
        '-c', '--correlation',
        action='store_true',
        help='If set, uses correlation coefficient analysis instead of '
             'a difference-based analysis.'
    )

    args = parser.parse_args()

    variability_curve = pd.read_csv(args.INPUT)
    event_boundaries = get_salience_indices()

    # All time points that are available and from which sampling is
    # possible. We do not account for the temporal boundaries of the
    # curve, though, so some events might not feature a proper window.
    possible_events = variability_curve.index.values

    evaluation_fn = get_variability_mean_difference
    if args.correlation:
        evaluation_fn = get_variability_correlation

    # Original estimate of the variability for the *true* event
    # boundaries.
    theta_0 = evaluation_fn(
        event_boundaries,
        variability_curve,
        args.window
    )

    non_events = sorted(set(possible_events) - set(event_boundaries))

    # Bootstrap distribution of the coefficient of determination.
    thetas = []

    n_bootstraps = 1000
    for i in tqdm(range(n_bootstraps), desc='Bootstrap'):
        m = len(event_boundaries)

        event_boundaries_bootstrap = np.random.choice(
            possible_events,
            m,
            replace=False  # to ensure that we do not get repeated events
        )

        event_boundaries_bootstrap = sorted(event_boundaries_bootstrap)

        thetas.append(
            evaluation_fn(
                event_boundaries_bootstrap,
                variability_curve,
                args.window
            )
        )

    print(theta_0)
    print(sum(thetas >= theta_0) / n_bootstraps)
    print(sum(thetas <= theta_0) / n_bootstraps)

    sns.distplot(thetas, bins=20)
    plt.show()
