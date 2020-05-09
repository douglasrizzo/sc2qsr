"""
Qualitative Spatial Reasoning -- :mod:`sc2qsr.spatial.qualitative`
******************************************************************
"""
from math import atan2, ceil, floor, pi
from numbers import Number

import numpy as np

import networkx as nx

from .quantitative import pol2cart
from scipy.spatial.distance import pdist, cdist
from matplotlib import pyplot as plt

DIRECTION_IDENTITY = -1
DISTANCE_IDENTITY = 0

# functions to convert between condensed and square matrix coordinates were taken from https://stackoverflow.com/a/36867493/1245214


def __calc_row_idx(k: int, n: int) -> int:
    """Calculates the column (x-coordinate) an element in position k of a condensed vector would be in its square matrix version

    :param k: index of element in the condensed vector
    :type k: int
    :param n: dimension of the square matrix
    :type n: int
    :return: x-coordinate of element k in the square matrix
    :rtype: int
    """
    return int(ceil((1 / 2.) * (-(-8 * k + 4 * n**2 - 4 * n - 7)**0.5 + 2 * n - 1) - 1))


def __elem_in_x_rows(x: int, n: int) -> int:
    return x * (n - 1 - x) + (x * (x + 1)) / 2


def __calc_col_idx(k: int, x: int, n: int) -> int:
    """Calculates the row (y-coordinate) an element in position k of a condensed vector would be in its square matrix version

    :param k: index of element in the condensed vector
    :type k: int
    :param x: x-coordinate of element in the squre matrix (see :py:func:`__calc_row_idx`)
    :type x: int
    :param n: dimension of the square matrix
    :type n: int
    :return: y-coordinate of element k in the square matrix
    :rtype: int
    """
    return int(n - __elem_in_x_rows(x + 1, n) + k)


def condensed_to_square(k: int, n: int) -> tuple:
    """Given the index k of an element in a condensed vector, returns its (x, y) coordinates in a corresponding square matrix. This function always returns coordinates from the lower triagular matrix (x > y).

    :param k: Index of desired element in the condensed matrix
    :type k: int
    :param n: dimension of the original square matrix
    :type n: int
    :return: tuple (x, y) containing coordinates of element in a square matrix
    :rtype: tuple
    """
    x = __calc_row_idx(k, n)
    y = __calc_col_idx(k, x, n)
    return x, y


def square_to_condensed(x: int, y: int, n: int) -> tuple:
    """Returns the index of an element in a condensed vector, given its (x,y) coordinates in a square matrix

    :param x: x coordinate of element
    :type x: int
    :param y: y coordinate of element
    :type y: int
    :param n: dimension of square matrix
    :type n: int
    :return: index of an element in the corresponding condensed vector
    :rtype: int
    """
    assert x != y, "no diagonal elements in condensed matrix"
    if x < y:
        x, y = y, x
    return int(n * y - y * (y + 1) / 2 + x - 1 - y)


def to_star(
    m: int, xa: float, ya: float, xb: float = 0, yb: float = 0, clockwise: bool = False
) -> int:
    """Calculates the :math:`STAR_m` relation of point :math:`A` wrt. to point :math:`B`. In the relation :math:`A(i)B`,

    .. math:: i = \\lfloor \\frac{\\arctan2(y_a - y_b, x_a - x_b) + \\pi}{2 \\pi} 2m \\mod 2m \\rfloor

    :param m: granularity of the STAR calculus
    :type m: int
    :param xa: x-coordinate of point A
    :type xa: float
    :param ya: y-coordinate of point A
    :type ya: float
    :param xb: x-coordinate of point B
    :type xb: float
    :param yb: y-coordinate of point B
    :type yb: float
    :param clockwise: whether angular sectors are numbered clockwise, defaults to False
    :type clockwise: bool, optional
    :return: STAR relation of point A wrt. to point B
    :rtype: int
    """

    x, y = xa - xb, ya - yb

    if x == 0 and y == 0:
        return DIRECTION_IDENTITY

    # atan2(ya - yb, xa - xb) = angle of A wrt. to B
    # add pi to make the result between 0 and 2 pi
    # divide by 2 pi to normalize angle from radians to [0; 1] interval
    # multiply by 2m, the number of angular sectors
    # mod (2m) to make the result < 2m
    # floor the result to transform the real result into the integer value related to the index of the angular STAR sector
    i = floor((((atan2(y, x) + pi) / (2 * pi)) * 2 * m) % (2 * m))

    # deal with both clockwise and anticlockwise sector numbering
    return i if not clockwise else int(2 * m - i)


def inverse(i: int, m: int) -> int:
    """Calculates the inverse of a relation on a :math:`STAR_m` calculus

    :param i: a relation
    :type i: int
    :param m: granularity of the STAR calculus
    :type m: int
    :return: the inverse of i
    :rtype: int
    """
    __validate_directions([i], m)
    return (i + m) % (2 * m)


def boundary_distance(i: int, elv: float, n: int) -> float:
    """Calculates the distance of the boundaries to an elevated origin point

    :param i: index of the boundary of interest
    :type i: int
    :param elv: elevation of the point
    :type elv: float
    :param n: distance granularity
    :type n: int
    :raises ValueError: if i is not even, it means it is not a boundary, but a sector, which is bounded by boundaries i - 1 and i + 1
    :return: Distance of boundary i to the origin point
    :rtype: float
    """
    if i > n * 2 or i < 0:
        raise ValueError(
            'Region index {} is either greater than 2n ({}) or less than 0'.format(i, 2 * n)
        )

    if i % 2 != 0:
        raise ValueError("region i is not a boundary (i must be even)")
    if i <= n:
        b = i * elv / n
    elif i < 2 * n:
        b = (n * elv) / (2 * n - i)
    else:
        b = float('inf')

    return b


def to_qdist(d: float, elevation: float, n: int) -> int:
    """Converts a quantitative distance to qualitative (boundary or sector)

    :param d: quantitative distance from the elevated point
    :type d: float
    :param elevation: elevation of point
    :type elevation: float
    :param n: distance granularity
    :type n: int
    :return: index of the qualitative distance region at distance d
    :rtype: int
    """
    if d < 0:
        raise ValueError('quantitative distance must not be negative')
    if elevation <= 0:
        raise ValueError('elevation must not less than or equal to 0')

    for i in range(2 * n):
        # i % 2 == 0 means the current region is a boundary
        if i % 2 == 0 and d == boundary_distance(i, elevation, n):
            return i
        elif i % 2 != 0 and boundary_distance(i - 1, elevation,
                                              n) < d < boundary_distance(i + 1, elevation, n):
            return i


def generate_qualitative_configuration(
    entities: np.ndarray, m: int = None, n: int = None, elevations=None
):
    if m is None and n is None:
        raise ValueError('Either m or n must be informed')
    if (n is None and elevations is not None) or (n is not None and elevations is None):
        raise ValueError(
            'Both n and elevations must be informed in order to calculate elevated distances'
        )

    if n is not None:
        quali_dists = generate_qualitative_distances(entities, elevations, n)
    if m is not None:
        quali_dirs = generate_qualitative_directions(entities, m)

    if n is None:
        return quali_dirs
    if m is None:
        return quali_dists

    return quali_dirs, quali_dists


def generate_qualitative_directions(entities: np.ndarray, m: int) -> np.ndarray:
    """Creates a condensed vector of qualitative spatial relations of direction between entities

    :param entities: an n x 2 array, where n is the number of entities and the columns represent the (x, y) doordinates of each entity
    :type entities: numpy.ndarray
    :param m: granularity parameter, m >= 2
    :type m: int
    :return: a condensed vector of qualitative spatial directions between all entities
    :rtype: numpy.ndarray

    .. tip:: Take a look at :func:`sc2qsr.spatial.qualitative.square_to_condensed` for how to convert :math:`(x, y)` coordinates into an ordinary square matrix into an index :math:`k`, where the direction of element :math:`y` w.r.t. element :math:`x` is stored in the condensed vector.
    """
    if m < 2:
        raise ValueError('the direction granularity parameter must be >= 2')

    n_entities = entities.shape[0]
    quali_dirs = np.zeros(int((n_entities**2 - n_entities) / 2), dtype=int)
    q_vector_index = -1
    for i in range(n_entities):
        # the very last entity is already processed, so we bail out
        if i == n_entities - 1:
            break

        for j in range(i + 1, n_entities):
            q_vector_index += 1
            quali_dirs[q_vector_index] = to_star(
                m, entities[i][0], entities[i][1], entities[j][0], entities[j][1]
            )

    return quali_dirs


def generate_qualitative_distances(entities: np.ndarray, elevations: list, n: int) -> np.ndarray:
    if n < 2:
        raise ValueError('the distance granularity parameter must be >= 2')

    n_entities = entities.shape[0]

    # if an integer is passed as elevation, a list is created with redundant values for all entities
    if isinstance(elevations, Number):
        elevations = [elevations] * n_entities

    if any((i <= 0 for i in elevations)):
        raise ValueError('There are elevation values <= 0')

    quali_dists = np.zeros((n_entities, n_entities), dtype=int)
    quant_dists = pdist(entities)

    # initialize the diagonal with the identity
    np.fill_diagonal(quali_dists, DISTANCE_IDENTITY)

    for i in range(n_entities - 1):
        for j in range(i + 1, n_entities):
            quant_dist = quant_dists[square_to_condensed(i, j, n_entities)]

            quali_dists[i, j] = to_qdist(quant_dist, elevations[i], n)
            quali_dists[j, i] = to_qdist(quant_dist, elevations[j], n)

    return quali_dists


def __validate_directions(qdir_vector: np.ndarray, m: int):
    """Checks if a condensed qualitative direction vector is consistent with a given qualitative direction granularity parameter

    :param qdir_vector: condensed qualitative direction vector
    :type qdir_vector: numpy.ndarray
    :param m: qualitative direction granularity parameter
    :type m: int
    :raises ValueError: if the condensed vector is not consistent with the granularity parameter
    """
    smaller = np.sum(np.nonzero((qdir_vector < 0) & (qdir_vector != DIRECTION_IDENTITY)))
    greater = np.sum(np.nonzero(qdir_vector >= 2 * m))

    error = ''

    if greater > 0:
        error = 'There are {} values greater than 2m'.format(greater)
    if smaller > 0:
        if len(error) == 0:
            error = 'There are '
        else:
            error = ' and '
        error += '{} values less than 0 and different than the direction identity value, which is {}'.format(
            smaller, DIRECTION_IDENTITY
        )

    if len(error) > 0:
        raise ValueError(error)


def qdir_squareform(qdir_vector: list, m: int) -> np.ndarray:
    """Converts a vector-form qualitative direction vector to a square-form qualitative direction matrix.

    :param qdir_vector: condensed qualitative direction vector
    :type qdir_vector: list
    :param m: direction granurality parameter
    :type m: int
    :raises ValueError: if qdir_vector has the wrong size to be transformed into a square matrix
    :return: square matrix with qualitative directions in one triangular, their inverses in the opposite triangular and the identity in the diagonal
    :rtype: np.ndarray
    """
    __validate_directions(qdir_vector, m)

    # Grab the closest value to the square root of the number
    # of elements times 2 to see if the number of elements
    # is indeed a binomial coefficient.
    dim = int(np.ceil(np.sqrt(len(qdir_vector) * 2)))

    # Check that v is of valid dimensions.
    if dim * (dim - 1) / 2 != int(len(qdir_vector)):
        raise ValueError(
            'Incompatible vector size. It must be a binomial coefficient n choose 2 for some integer n >= 2.'
        )

    q_matrix = np.zeros((
        dim,
        dim,
    ), dtype=int)

    for i in range(len(qdir_vector)):
        x, y = condensed_to_square(i, dim)
        q_matrix[x, y] = qdir_vector[i]
        q_matrix[y, x] = inverse(qdir_vector[i], m)

    np.fill_diagonal(q_matrix, DIRECTION_IDENTITY)

    return q_matrix


def qualitative_with_reference(
    a: np.ndarray, m: int, n: int, elevation: float, ref: np.ndarray = np.zeros((1, 2))
) -> np.ndarray:
    if not isinstance(ref, np.ndarray):
        ref = np.array(ref)
    if ref.shape != (1, 2):
        ref = ref.reshape((1, 2))

    quant_dists = cdist(a, ref)

    p = np.zeros((a.shape[0], 2), dtype=int)

    for i, row in enumerate(a):
        p[i, 0], p[i, 1] = to_qdist(quant_dists[i], elevation,
                                    n), to_star(m, row[0], row[1], ref[0, 0], ref[0, 1])

    return p


def epra2pol(
    qdir: int,
    qdist: int,
    m: int,
    n: int,
    elevation: float = None,
    clockwise: bool = False
) -> tuple:
    """Returns approximate polar coordinates for a qualitative sector of direction and distance

    :param qdir: qualitative direction
    :type qdir: int
    :param qdist: qualitative distance
    :type qdist: int
    :param m: qualitative direction granularity parameter
    :type m: int
    :param n: qualitative distance granularity parameter
    :type n: int
    :param elevation: elevation of the point, defaults to None
    :type elevation: float, optional
    :param clockwise: whether to generate coordinates in a clockwise fashion, defaults to False
    :type clockwise: bool, optional
    :return: a tuple containing (rho, phi), that is, distance and angle
    :rtype: tuple
    """
    angular_sector_size = (2 * pi) / (2 * m)
    phi = angular_sector_size * qdir + (pi / 2)

    if clockwise:
        phi = pi - phi

    if elevation is None:
        rho = qdist
    else:
        # qdist % 2 == 0 means the current region is a boundary
        if qdist % 2 == 0:
            rho = boundary_distance(qdist, elevation, n)
        elif qdist % 2 != 0:
            rho = (
                boundary_distance(qdist - 1, elevation, n) +
                boundary_distance(qdist + 1, elevation, n)
            ) / 2

    return rho, phi


def epra2cart(
    qdir: int,
    qdist: int,
    m: int,
    n: int,
    elevation: float = None,
    clockwise: bool = False
) -> tuple:
    """Returns an approximate cartesian position for a qualitative sector of direction and distance

    :param qdir: qualitative direction
    :type qdir: int
    :param qdist: qualitative distance
    :type qdist: int
    :param m: qualitative direction granularity parameter
    :type m: int
    :param n: qualitative distance granularity parameter
    :type n: int
    :param elevation: elevation of the point, defaults to None
    :type elevation: float, optional
    :param clockwise: whether to generate coordinates in a clockwise fashion, defaults to False
    :type clockwise: bool, optional
    :return: a tuple containing x, y coordinates
    :rtype: tuple
    """
    rho, phi = epra2pol(qdir, qdist, m, n, elevation, clockwise)

    return pol2cart(rho, phi)


def normalize_direction(qdir_vector: np.ndarray, m: int) -> np.ndarray:
    """Normalize a condensed qualitative direction vector so that the relative position between all entities does not affect its final representation. This is done by rotating all relations until the first relation in the vector equals 0.

    :param qdir_vector: condensed qualitative direction vector
    :type qdir_vector: numpy.ndarray
    :param m: qualitative direction granularity parameter, >= 2
    :type m: int
    :return: a normalized version of `qdir_vector`
    :rtype: numpy.ndarray
    """
    __validate_directions(qdir_vector, m)

    # find the first value that is not the identity
    index = np.argmax(qdir_vector != DIRECTION_IDENTITY)
    v = qdir_vector[index]

    # if that value is already 0, do nothing
    if v == 0:
        return qdir_vector

    # perform the rotation in all relations, making the first non-identity relation = 0
    new_dirs = (qdir_vector - v) % (2 * m)

    # get all positions in the original vector that
    # contained the identity and reassign them
    eq = np.nonzero(qdir_vector == DIRECTION_IDENTITY)
    new_dirs[eq] = DIRECTION_IDENTITY

    return new_dirs


def equals_absolute_direction(qdv1: np.ndarray, qdv2: np.ndarray, m: int) -> bool:
    """Checks if two condensed qualitative direction vectors are equivalent. This is done by normalizing both vectors and checking if they are equal afterwards.

    :param qdv1: first condensed qualitative direction vector
    :type qdv1: numpy.ndarray
    :param qdv2: secondcondensed qualitative direction vector
    :type qdv2: numpy.ndarray
    :param m: qualitative direction granularity parameter
    :type m: int
    :return: True if `qdv1` and `qdv2` are equivalent, else `False`
    :rtype: bool
    """
    __validate_directions(qdv1, m)
    __validate_directions(qdv2, m)
    return np.array_equal(normalize_direction(qdv1, m), normalize_direction(qdv2, m))


def create_cnd_graph(m: int, n: int, clockwise: bool = False):
    G = nx.Graph()

    for i in range(2 * m):
        next_i = (i + 1) % (2 * m)
        for j in range(1, 2 * n - 1):
            G.add_edge((i, j), (next_i, j))
            G.add_edge((i, j), (i, j + 1))

    for i in range(2 * m):
        next_i = (i + 1) % (2 * m)
        G.add_edge((DIRECTION_IDENTITY, DISTANCE_IDENTITY), (i, DISTANCE_IDENTITY + 1))
        G.add_edge((i, 2 * n - 1), (next_i, 2 * n - 1))

    return G


def draw_cnd_graph(m: int, n: int, dimensions: tuple = (12., 12.), clockwise: bool = False):

    G = create_cnd_graph(m, n, clockwise)

    positions = {}
    positions[(DIRECTION_IDENTITY, DISTANCE_IDENTITY)] = (0, 0)
    for i in range(2 * m):
        for j in range(1, 2 * n):
            positions[(i, j)] = epra2cart(i, j, m, n, None, clockwise)

    plt.figure(figsize=dimensions)
    nx.draw(G, positions, with_labels=True, font_weight='bold')


def create_cnd_matrix(G: nx.Graph, m: int, n: int):
    pass