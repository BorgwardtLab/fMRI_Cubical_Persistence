'''
Module containing functions to compute curvature
'''

import numpy as np
from scipy.spatial.distance import euclidean as euclid

def arc_chord_ratio(vals):
    '''
    Computes :math:`\frac{C}{L}` where :math:`C` is the sum over the euclidean
    distance between subsequent points and :math:`L` the euclidean distance
    between start and end point.

    Parameters
    ---------

        vals (list): Matrix of dimension T times n
    
    Returns
    -------

    Tortuosity of points that describe a path.
    '''
    individual_dist = [euclid(vals[i-1], vals[i]) for i in range(1, len(vals))]
    C = np.sum(individual_dist)
    L = euclid(vals[0], vals[-1])
    return C / L

