'''
Module containing utility functions.
'''

import os
import re
from collections import defaultdict
from glob import glob


def parse_filename(filename):
    '''
    Parses a filename and composes it into different components.
    Depending on the filename, the following information can be
    extracted:

    - Subject ID
    - Task (abbreviation)
    - Time step (optional)

    These items will be returned as tuples. If time information
    is not available, `None` will be returned.
    '''

    filename = os.path.splitext(filename)[0]
    tokens = filename.split('_')

    for token in tokens:
        if 'sub' in token:
            subject = re.match('\D*(\d+)\D*', token).group(1)
        elif 'task' in token:
            task = token.split('-')[1]

    potential_time = tokens[-1]

    if re.match('\d+', potential_time):
        time = potential_time
    else:
        time = None

    return subject, task, time

def get_patient_ids_and_times(path_to_data: str, task: str='pixar'):
    '''
    Returns a list of patient ids and available time steps
    based on the data available.
    '''
    
    all_patients = defaultdict(list)
    for f in glob(os.path.join(path_to_data, f'sub-{task}*_task-{task}_bold_space-MNI152NLin2009cAsym_preproc_*.json')):
        subject, task, time = parse_filename(f)
        all_patients[subject].append(time)

    all_patients = {k: sorted(v) for k, v in all_patients.items()}
    sorted_patients = {}
    for k in sorted(all_patients):
        sorted_patients[k] = all_patients[k]

    return sorted_patients

