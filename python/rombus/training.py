# --- training.py ---

"""
    Classes for building training sets
"""

__author__ = "Chad Galley <crgalley@gmail.com>"

import random

import numpy as np
from numpy import asarray


##############################################
class TrainingTools:
    def __init__(self):
        pass

    def equal_lens(self, *arrs):
        lens = map(len, arrs)
        if len(set(lens)) > 1:
            return False
        else:
            return True

    def check_lens(self, *arrs):
        if not self.equal_lens(*arrs):
            raise Exception("Unequal array lengths")
        pass

    def make_arrs(self, pmin, pmax, Nmu):
        """Make list of 1d arrays for each given parameter range"""
        if np.shape(Nmu):
            return [np.linspace(pmin[jj], pmax[jj], Nmu[jj]) for jj in range(len(pmin))]
        else:
            return [np.linspace(pmin[jj], pmax[jj], Nmu) for jj in range(len(pmin))]

    def meshgrid2(self, *arrs):
        """Multi-dimensional version of numpy's meshgrid"""
        arrs = tuple(reversed(arrs))
        lens = map(len, arrs)
        dim = len(arrs)
        sz = 1
        for s in lens:
            sz *= s

        ans = []
        for i, arr in enumerate(arrs):
            slc = [1] * dim
            slc[i] = lens[i]
            arr2 = asarray(arr).reshape(slc)
            for j, sz in enumerate(lens):
                if j != i:
                    arr2 = arr2.repeat(sz, axis=j)
            ans.append(arr2)

        return ans[::-1]

    def meshgrid3(self, *arrs):
        """Multi-dimensional version of numpy's meshgrid,
        excluding symmetric tuples from the square grid"""
        arrs = tuple(reversed(arrs))
        lens = map(len, arrs)
        inds = [np.triu_indices(ll) for ll in lens]
        dim = len(arrs)
        sz = 1
        for s in lens:
            sz *= s

        ans = []
        for i, arr in enumerate(arrs):
            slc = [1] * dim
            slc[i] = lens[i]
            arr2 = asarray(arr).reshape(slc)
            for j, sz in enumerate(lens):
                if j != i:
                    arr2 = arr2.repeat(sz, axis=j)
            ans.append(arr2[inds[i]])

        return ans[::-1]

    def tuple_to_vstack(self, arr):
        return np.vstack(map(np.ravel, tuple(arr)))

    def rescale(self, x):
        """Scale [a,b] to [0,1]"""
        a = x[0]
        b = x[-1]
        return (x - a) / (b - a)

    def invrescale(self, x, a, b):
        """Scale [0,1] to [a,b]"""
        if float(x[0]) != 0.0 or float(x[-1]) != 1.0:
            raise Exception("Expected a [0,...,1] array")
        return a + (b - a) * x


##############################################
class Uniform(TrainingTools):
    def __init__(self):
        TrainingTools.__init__(self)
        self.switch_dict = {
            "grid": self.grid,
            "symgrid": self.sym_grid,
            "1d": self.oned,
            "1d-constrained": self.oned_constrained,
        }
        self.options = self.switch_dict.keys()

    def __call__(self, params, training_type):
        return self.switch_dict[training_type](*params)

    def oned(self, pmin, pmax, Nmu):
        """Uniformly spaced 1d array"""
        return np.linspace(pmin, pmax, num=Nmu)

    def oned_constrained(self, pmin, pmax, Nmu):
        """Uniformly spaced 1d array satisfying a constraint"""
        arrs = self.make_arrs(pmin, pmax, Nmu)
        return self.tuple_to_vstack(arrs)

    def grid(self, pmin, pmax, Nmu):
        """Uniformly spaced n-dimensional array"""
        arrs = self.make_arrs(pmin, pmax, Nmu)
        return self.tuple_to_vstack(self.meshgrid2(*arrs))

    def sym_grid(self, pmin, pmax, Nmu):
        """Uniformly spaced n-dimensional array with
        symmetric tuples removed from square grid
        """
        arrs = self.make_arrs(pmin, pmax, Nmu)
        if np.shape(Nmu):
            if len(set(Nmu)) > 1:
                raise Exception(
                    "Number of points in each parameter dimension are not equal."
                )
        return self.tuple_to_vstack(self.meshgrid3(*arrs))


##############################################
class Random(TrainingTools):
    def __init__(self):
        TrainingTools.__init__(self)
        self.switch_dict = {"uniform": self.uniform, "nonuniform": self.nonuniform}
        self.options = self.switch_dict.keys()

    def __call__(self, params, training_type):
        return self.switch_dict[training_type](*params)

    def uniform(self, pmin, pmax, Nmu):
        if np.shape(pmin):
            lens = map(len, [pmin, pmax])
            ans = []
            if self.equal_lens(pmin, pmax):
                for ii in range(lens[0]):
                    arr = [random.uniform(pmin[ii], pmax[ii]) for jj in range(Nmu)]
                    ans.append(arr)
            return self.tuple_to_vstack(ans)
        else:
            return self.tuple_to_vstack(
                [[random.uniform(pmin, pmax)] for jj in range(Nmu)]
            )

    def nonuniform(self):
        pass


##############################################
class NonUniform(TrainingTools):
    def __init__(self):
        TrainingTools.__init__(self)
        self.switch_dict = {"1d": self.oned, "1d-constrained": self.oned_constrained}
        self.options = self.switch_dict.keys()

    def __call__(self, fn, params, training_type):
        return self.switch_dict[training_type](fn, *params)

    def oned(self, fn, pmin, pmax, Nmu):
        """Nonuniform 1d training space made from an input function"""
        x = np.linspace(pmin, pmax, num=Nmu)
        u = fn(self.rescale(x))
        return self.invrescale(u, pmin, pmax)

    def oned_constrained(self, fn, pmin, pmax, Nmu):
        """Nonuniform 1d, constrained training space made from an input function"""
        mu = np.transpose([pmin, pmax, Nmu])
        arrs = []
        for mm in mu:
            arrs.append(self.oned(fn, *mm))
        return self.tuple_to_vstack(arrs)
