#!/usr/bin/env python3
#
# Performs peri-event histogram analysis for a variability curve. This
# code was kindly contributed by Tristan Yates.

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from scipy.stats import sem


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')

    args = parser.parse_args()

    variability_curve = pd.read_csv(args.INPUT)
    event_boundaries = get_salience_indices()

    # This is the maximum time stored in the data set; it does not
    # necessarily correspond to the length of the curve because it
    # is possible that indices have been dropped.
    T = variability_curve['time'].max()

    # Makes it easier to access a given time step; note that we are
    # using the indices from the event boundaries here.
    variability_curve.set_index('time', inplace=True)

    # Make the peri-event curve
    w = 3
    peri_event = np.zeros((w*2 + 1, len(event_boundaries)))

    for idx, t in enumerate(range(-w, w+1)):
        for eb, bound in enumerate(event_boundaries):
            if bound + t < 0 or bound + t > T:
                peri_event[idx, eb] = np.nan
            else:
                peri_event[idx, eb] = variability_curve.loc[bound + t]

    name = os.path.splitext(os.path.basename(args.INPUT))[0]
    plt.title(name)

    colors = cm.coolwarm(np.linspace(0, 1, peri_event.shape[0]))
    plt.vlines(0,np.nanmin(peri_event),np.nanmax(peri_event),linestyle='dashed')
    plt.bar(np.arange(-w,w+1),np.nanmean(peri_event,axis=1),yerr=sem(peri_event, axis=1, nan_policy='omit'),color=colors)
    plt.ylim(np.nanmin(peri_event),np.nanmax(peri_event))

    plt.show()
