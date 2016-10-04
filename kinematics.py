#!bin/usr/env python

import numpy as np
from scipy import integrate
"""
Taking a set of accelerometer data, integrate each component independently to get an array of velocity and position values.
"""

def velocity(t,a):
    """
    First integral to find velocity from acceleration.

    Parameters
    ----------
        t: panda.core.series.Series
            time values for accelerometer data
        a: panda.core.series.Series
            accelerometer data

    Returns
    -------
        v: nd.array
            array of velocity values
    """
    v = integrate.cumtrapz(a,t,initial = 0)
    return v

def test_velocity():
    """
    Tests velocity() for t = (0,1,2,3), a = (1,4,7,10)
    """
    t = np.array([0.0,1.0,2.0,3.0])
    a = np.array([1.0,4.0,7.0,10.0])
    test = velocity(t,a)
    case = np.array([0.000,0.000,2.500,8.000])
    success = 0

    def a_eq(a,b,eps):
        return (abs(a - b) < eps)

    for (j,k) in zip(test,case):
        if (a_eq(j,k,1)):
            success += 1
    assert (success == 4)
