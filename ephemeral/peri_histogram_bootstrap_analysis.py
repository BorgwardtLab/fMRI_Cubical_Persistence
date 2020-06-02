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
from scipy.stats import pearsonr

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

    clf = LinearRegression()
    clf.fit(X, y)

    return pearsonr(y.ravel(), clf.predict(X).ravel())[0]


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


def get_variability_max_difference(events, curve, w):
    """Calculate max difference in pre-event and post-event variability.

    The basic idea behind this function is to take a set of events, or
    rather event boundaries, and evaluate the variability of the curve
    in terms of its variability over a window pre-event and post-event.

    Variability is assessed by calculating the differences between the
    minimum and the maximum on both sides of the boundary. Again, this
    function can easily be used in for bootstrapping by using a random
    selection of events.

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
    Difference in variabilities pre-event and post-event.
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

    # Calculate pre-event variability and post-event variability. Notice
    # that we are not including the event point itself.
    pre_event_variability = np.max(means[:w]) - np.min(means[:w])
    post_event_variability = np.max(means[w+1:]) - np.min(means[w+1:])

    return pre_event_variability - post_event_variability


def get_all_shifts(event_boundaries, variability_curve):
    """Return all potential shifts of event boundaries."""
    T = np.max(variability_curve.index.values)

    all_boundaries_shifted = []

    for shift in range(T):
        all_boundaries_shifted.append(np.mod(event_boundaries + shift, T))

    return all_boundaries_shifted


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
        '-b', '--bootstrap',
        default=1000,
        type=int,
        help='Specifies number of bootstrap samples'
    )

    parser.add_argument(
        '-c', '--correlation',
        action='store_true',
        help='If set, uses correlation coefficient analysis instead of '
             'a difference-based analysis.'
    )

    parser.add_argument(
        '-s', '--shift',
        action='store_true',
        help='If set, evaluates all potential event shifts instead of '
             'performing a bootstrap-based analysis.'
    )

    args = parser.parse_args()

    variability_curve = pd.read_csv(args.INPUT)
    event_boundaries = get_salience_indices()

    # All time points that are available and from which sampling is
    # possible. We do not account for the temporal boundaries of the
    # curve, though, so some events might not feature a proper window.
    possible_events = variability_curve.index.values

    evaluation_fn = get_variability_max_difference
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

    # Shift-based analysis
    if args.shift:
        all_boundaries_shifted = get_all_shifts(
            event_boundaries,
            variability_curve
        )

        # Stores the relevant values for each shift; this makes it
        # easier to assess the effects in a visualisation later on
        thetas = np.zeros((len(all_boundaries_shifted) + 1, 1))
        thetas[0] = theta_0

        # Counts how many events there are in total for each of the time
        # points.
        n_events = [len(event_boundaries)]

        for index, boundaries in enumerate(all_boundaries_shifted):
            boundaries = sorted(boundaries)

            print(index + 1, (event_boundaries - boundaries).sum())

            n_events.append(np.sum(np.isin(boundaries, event_boundaries)))

            thetas[index + 1] = evaluation_fn(
                boundaries,
                variability_curve,
                args.window
            )

        print(theta_0)
        print(sum(thetas > theta_0) / thetas.shape[0])
        print(sum(thetas < theta_0) / thetas.shape[0])

        sns.lineplot(np.arange(thetas.shape[0]), thetas.ravel())

    # Bootstrap analysis
    else:
        n_bootstraps = args.bootstrap

        # Will contain the bootstrap distribution in this case, making
        # it possible to assess to what extent extreme values are likely
        # to appear.
        thetas = []

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
        print(sum(thetas > theta_0) / n_bootstraps)
        print(sum(thetas < theta_0) / n_bootstraps)

        sns.distplot(thetas, rug=True)

    plt.show()
