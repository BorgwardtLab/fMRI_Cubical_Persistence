'''
Module containing functions to compute curvature
'''

import numpy as np
from IPython import embed
from scipy.spatial.distance import euclidean as euclid

def arc_chord_ratio(vals, return_ind=False):
    '''
    Computes :math:`\frac{C}{L}` where :math:`C` is the sum over the euclidean
    distance between subsequent points and :math:`L` the euclidean distance
    between start and end point.

    Parameters
    ---------

        vals (list): Matrix of dimension T times n
        return_ind (bool): Whether to return individual distances and last dist
    
    Returns
    -------

        Tortuosity of points that describe a path.
    '''
    individual_dist = [euclid(vals[i-1], vals[i]) for i in range(1, len(vals))]
    C = np.sum(individual_dist)
    L = euclid(vals[0], vals[-1])

    if return_ind:
        return C / L, individual_dist, L
    else:
        return C / L

def curve_properties(vals):
    '''
    Computes curve properties such as `speed`, `velocity`, `curvature`, and 
    `acceleration`. 

    Parameters
    ---------

        vals (list): Matrix of dimension T times 2

    Returns
    -------
        
        velocity (list): T times 2 array with velocity for each component
        speed (list): T times 1 vector with speed at each time point 
        curvature (list): T times 1 vector with curvature at each time point
        acceleration (list): T times 2 array with acceleration for each
        component
    '''

    # Compute velocity
    dx_dt = np.gradient(vals[:, 0])
    dy_dt = np.gradient(vals[:, 1])
    velocity = np.array([[dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])

    # Compute speed
    ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)

    # Compute curvatue
    d2s_dt2 = np.gradient(ds_dt)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    curvature = np.abs((d2x_dt2 * dy_dt) - (dx_dt * d2y_dt2)) / ((dx_dt * dx_dt)
                                                                 + (dy_dt
                                                                    * dy_dt))**1.5

    tangent = np.array([1/ds_dt]).transpose() * velocity
    tangent_x = tangent[:, 0]
    tangent_y = tangent[:, 1]

    deriv_tangent_x = np.gradient(tangent_x)
    deriv_tangent_y = np.gradient(tangent_y)

    dT_dt = np.array([ [deriv_tangent_x[i], deriv_tangent_y[i]] for i in range(deriv_tangent_x.size)])

    length_dT_dt = np.sqrt(deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y)

    normal = np.array([1/length_dT_dt]).transpose() * dT_dt

    t_component = np.array([d2s_dt2] * 1).transpose()
    n_component = np.array([curvature * ds_dt * ds_dt]).transpose()
    
    # Compute acceleration
    acceleration = t_component * tangent + n_component * normal

    return velocity, ds_dt, curvature, acceleration


