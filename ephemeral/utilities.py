'''
Module containing utility functions.
'''

import re


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

    tokens = filename.split('_')

    for token in tokens:
        if 'sub' in token:
            subject = re.match('\D*(\d+)\D*', token).group(1)
        elif 'task' in token:
            task = token.split('-')[1]

    potential_time = tokens[-1]

    if re.match('\d+', potential_time):
        time = potential_time[:-4]
    else:
        time = None

    return subject, task, time
