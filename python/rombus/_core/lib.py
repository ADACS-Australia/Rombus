# -- lib.py --

"""Basic functions used by several subpackages."""

__author__ = "Chad Galley <crgalley@gmail.com>"

import numpy as np
from scipy.special import factorial


def malloc(dtype, *nums):
    """Allocate some memory with given dtype"""
    return np.zeros(tuple(nums), dtype=dtype)


def malloc_more(arr, num_more):
    """Allocate more memory to append to arr"""
    dim = len(arr.shape)
    if dim == 1:
        return np.hstack([arr, malloc(arr.dtype, num_more)])
    elif dim == 2:
        # Add num_extra rows to arr
        shape = arr.shape
        return np.vstack([arr, malloc(arr.dtype, num_more, shape[1])])
    else:
        raise Exception("Expected a vector or matrix.")


def trim(arr, num):
    return arr[:num]


def scale_ab_to_cd(x, c, d):
    """Scale [a,b] to [c,d]"""
    a = x[0]
    b = x[-1]
    a, b, c, d = map(float, [a, b, c, d])
    return (d - c) / (b - a) * x - (a * d - b * c) / (b - a)


def scale_ab_to_01(x):
    """Scale [a,b] to [0,1]"""
    interval = scale_ab_to_cd(x, 0, 1)
    return np.abs(interval)


def scale_01_to_ab(x, a, b):
    """Scale [0,1] to [a,b]"""
    if np.allclose(float(x[0]), 0.0) and np.allclose(float(x[-1]), 1.0):
        return scale_ab_to_cd(np.abs(x), a, b)
    else:
        raise Exception("Expected a [0,...,1] array")


def fneval(x, fn, args=None):
    if args is None:
        return fn(x)
    elif len(args) >= 1:
        return fn(x, *args)
    else:
        raise Exception("Function arguments not a 1d array.")
    pass


def get_arg(a, a0):
    """Get argument at which a0 occurs in array a"""
    return abs(a - a0).argmin()


def map_intervals(x, a, b):
    """Map array x to interval [a,b]"""
    M = (b - a) / (np.max(x) - np.min(x))
    B = a - M * np.min(x)
    return M * x + B


def choose(top, bottom):
    """Combinatorial choose function"""
    return factorial(top) / factorial(bottom) / factorial(top - bottom)


class Empty:
    """A class with no purpose other than for attaching class attributes"""

    def __init__(self):
        pass
