"""
Quantitative Spatial Functions -- :mod:`sc2qsr.spatial.quantitative`
********************************************************************
"""
import numpy as np
from scipy.spatial.distance import pdist
from math import atan2, sqrt


def cart2pol(x: float, y: float) -> tuple:
    """Convert cartesian coordinates to polar

    :param x: cartesian coordinate in the X direction
    :type x: float
    :param y: cartesian coordinate in the Y direction
    :type y: float
    :return: a tuple containing the euclidean distance of point (x,y) and its angle in radians, w.r.t. to the origin
    :rtype: tuple
    """
    return (sqrt(x**2 + y**2), atan2(y, x))


def pol2cart(rho: float, phi: float) -> tuple:
    """Convert polar coordinates to cartesian

    :param rho: distance of the point to the origin
    :type rho: float
    :param phi: angle of the point wrt. to the origin
    :type phi: float
    :return: a tuple containing the (x,y) coordinates of the point w.r.t. the origin
    :rtype: tuple
    """
    return (rho * np.cos(phi), rho * np.sin(phi))


def generate_polar_configuration(a: np.ndarray):
    # TODO must test output of this
    distances = pdist(a)

    angles = np.array(
        [atan2(a[i, 1] - a[i + 1, 1], a[i, 0] - a[i + 1, 0]) for i in range(a.shape[0] - 1)]
    )

    return angles, distances


def polar_with_reference(a: np.ndarray, ref: tuple = (0, 0)) -> np.ndarray:
    p = np.empty((a.shape[0], 2), dtype=float)
    for i, row in enumerate(a):
        p[i, 0], p[i, 1] = cart2pol(row[0] - ref[0], row[1] - ref[1])

    return p
