#!/usr/bin/env python3
#
# Performs peri-event histogram analysis for a variability curve. This
# code was kindly contributed by Tristan Yates.

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def get_salience_indices():
    """Return indices with salient events."""
    df_annot = pd.read_excel('../data/annotations.xlsx')

    # Get the salience values; it is perfectly justified to replace NaN
    # values by zero because those salience values will not be counted.
    salience = df_annot['Boundary salience (# subs out of 22)'].values
    salience = np.nan_to_num(salience)

    # These are the detected event boundaries, according to Tristan's
    # analysis. Note that since the index has been shifted above, the
    # dropping operation does *not* have to be considered here!
    salience_indices, = np.nonzero(salience >= 7)

    # Again, no shift here!
    return salience_indices


if __name__ == '__main__':

    # Get the variability data
    which_type='individual'
    smooth='yes'
    high_sal='no'

    if which_type=='individual':
        # All subs
        mask='occipitalmask' # occipitalmask or brainmask
        measure='summary_statistics' # persistence_images or summary_statistics
       
        if smooth=='yes':
            variability_time=pd.read_csv('variability/Variability_%s_%s_smooth3.csv' % (mask,measure))
        else:
            variability_time=pd.read_csv('variability/Variability_%s_%s.csv' % (mask,measure))
    elif which_type=='cohort':
        mask='brainmask'
        measure='cohort_r3' # persistence_images or summary_statistics

        variability_time=pd.read_csv('variability/%s_%s.csv' % (mask,measure))

    variability_time.head()

    # Get the boundaries 
    event_boundaries=np.load('event_boundaries_not_shifted.npy') #starts at first movie time point, not at blank period
    print('Event boundaries:',event_boundaries)

    highsal_event_boundaries=np.load('highsal_event_boundaries_not_shifted.npy')
    print('Most salient event boundaries:',highsal_event_boundaries)

    # Do you want the highly salient events or all event boundaries?
    if high_sal=='yes':
        test_boundaries=highsal_event_boundaries
    else:
        test_boundaries=event_boundaries

    # Make the peri-event curve
    peri_event=np.zeros((w*2+1,len(test_boundaries)))

    for idx,t in enumerate(range(-w,w+1)):
       
        for eb, bound in enumerate(test_boundaries):
           
            if bound+t<0 or bound+t > len(variability_shifted)-1:
                peri_event[idx,eb]=np.nan
            else:
                peri_event[idx,eb]=variability_shifted[bound+t]

    plt.title('%s %s' %(mask,measure))
    colors = cm.coolwarm(np.linspace(0, 1, peri_event.shape[0]))
    plt.vlines(0,np.nanmin(peri_event),np.nanmax(peri_event),linestyle='dashed')
    plt.bar(np.arange(-w,w+1),np.nanmean(peri_event,axis=1),yerr=stats.sem(peri_event,axis=1,nan_policy='omit'),color=colors)
    plt.ylim(np.nanmin(peri_event),np.nanmax(peri_event))
