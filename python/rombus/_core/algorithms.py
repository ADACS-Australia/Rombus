# --- algorithms.py ---

"""
Standard algorithms for::
   -- generating an orthonormal basis from a set of functions
   -- building a reduced spline for 1d functions (removed)
   -- building a reduced basis
   -- empirical interpolation
   -- reduced-order quadrature rules (future)
"""

__author__ = "Chad Galley <crgalley@gmail.com>"

import time

import numpy as np

from . import eim
from . import greedy
from . import lib


##############################################
class Basis(greedy.IteratedModifiedGramSchmidt):
    def __init__(self, hs, inner, inner_type, normsQ=False, dtype="complex"):
        greedy.IteratedModifiedGramSchmidt.__init__(self, inner, inner_type)

        self.Nbasis, self.Nquads = np.shape(hs)
        self.functions = hs

        self.normsQ = normsQ
        if self.normsQ:
            self.norms = lib.malloc(dtype, self.Nbasis)

        self.basis = lib.malloc(dtype, self.Nbasis, self.Nquads)
        pass

    def iter(self, step, h, a=0.5, max_iter=3):
        ans = self.add_basis(h, self.basis[:step], a=a, max_iter=max_iter)

        if self.normsQ:
            self.basis[step + 1], self.norms[step + 1] = ans
        else:
            self.basis[step + 1], _ = ans

        pass

    def make(self, a=0.5, max_iter=3, timerQ=False):
        """Find the corresponding orthonormal set of basis functions."""

        self.basis[0] = self.inner.normalize(self.functions[0], self.inner_type)
        if self.normsQ:
            self.norms[0] = self.inner.norm(self.functions[0], self.inner_type)

        if timerQ:
            t0 = time.time()

        for ii in range(1, self.Nbasis):
            self.iter(ii, self.functions[ii], a=a, max_iter=max_iter)

        if timerQ:
            print("\nElapsed time =", time.time() - t0)

        if self.normsQ:
            return [np.array(self.basis), self.norms]
        else:
            return np.array(self.basis)
        pass


##############################################
# class MakeRS(ReducedSpline1d, Spline1d):
#
#     def __init__(self, x, deg=3, dtype='double'):
#         ReducedSpline1d.__init__(self)
#         self.x = np.sort(x)
#         self.deg = deg
#         self.nodes = np.zeros(len(x), dtype='double')
#         self.values = np.zeros(len(x), dtype=dtype)
#         pass
#
#     def seed(self, x0, fn, args=None):
#         self.xmax = self.x.max()
#         self.nodes[0] = x0
#         self.values[0] = lib.fneval(x0, fn, args=args)
#         pass
#
#     def iter(self, step, delta, x, fn, args=None):
#         if x+delta < self.xmax:
#             self.nodes[step+1] = x+delta
#             self.values[step+1] = lib.fneval(x+delta, fn, args=args)
#             return x+delta, step+1
#         else:
#             self.nodes[step+1] = self.xmax
#             self.values[step+1] = lib.fneval(self.xmax, fn, args=args)
#             return self.xmax, step+1
#         pass
#
#     def make(self, tol, fn, args=None):
#
#         self.seed(self.x.min(), fn, args=args)
#
#         #xsorted = np.sort(self.x)
#         ctr = 0
#         xx = self.nodes[0]
#         while xx <= self.xmax:
#
#             deriv_n = self.autodiff(xx, fn, n=self.deg+1, args=args)
#             delta = self.delta(tol, deriv_n, self.deg+1)
#
#             xx, ctr = self.iter(ctr, delta, xx, fn, args=args)
#
#             if xx+delta > self.xmax:
#                 break
#
#         self.trim(ctr+1)
#         self.size = len(self.nodes)
#
#         # Make the spline interpolant
#         spline = Spline1d(self.nodes, self.values, k=self.deg)
#         self.fit = spline.fit
#         self.fitparams = spline.fitparams
#         pass
#
#     def trim(self, num):
#         """Trim zeros from remaining entries"""
#         self.nodes = self.nodes[:num+1]
#         self.values = self.values[:num+1]
#         pass


##############################################
class StandardRB(greedy.ReducedBasis, greedy.IteratedModifiedGramSchmidt):
    """Class for standard reduced basis greedy algorithm.

    Input
    -----
       Nbasis       -- number of basis functions
       Npoints       -- number of training set points
       Nquads       -- number of quadrature nodes used for inner products
       inner       -- method of InnerProduct instance
       inner_type -- type of inner product (e.g., 'complex')
       norms       -- array of training set function norms (default None)
       dtype       -- data type of functions (default 'complex')

    Functions
    ---------
       seed -- seed the greedy algorithm
       iter -- one iteration of the greedy algorithm
       make -- implement the greedy algorithm from beginning to end
       trim -- trim zeros from remaining allocated entries

    Examples
    --------
       Create an instance of the StandardRB class for functions with
       unit norm::

         >>> rb = rp.StandardRB(Nbasis, Npoints, Nquads, inner, inner_type)

       Let T be the training space of functions, 0 be the seed index,
       and 1e-12 be the tolerance. The standard reduced basis greedy
       algorithm for normalized functions is::

         >>> rb.seed(0, T)
         >>> for i in range(Nbasis):
         >>> ...if rb.errors[i] <= 1e-12:
         >>> ......break
         >>> ...rb.iter(i,T)
         >>> rb.trim(i)

       For convenience, this algorithm is equivalently implemented in
       `make`::

         >>> rb.make(0, T, 1e-12)

       Let T' be a different training space. The greedy algorithm can
       be run again on T' using::

         >>> rb.make(0, T', 1e-12)

       or, alternatively, at each iteration using::

         >>> ...rb.iter(i,T')

       in the for-loop above.

    """

    def __init__(
        self,
        Nbasis,
        Npoints,
        Nquads,
        inner,
        inner_type,
        Nmodes=1,
        norms=None,
        dtype="complex",
    ):
        greedy.ReducedBasis.__init__(self, inner, inner_type)
        greedy.IteratedModifiedGramSchmidt.__init__(self, inner, inner_type)

        if norms is None:
            self.norms = np.ones(Npoints, dtype="double")
        else:
            self.norms = norms
        self.ctr = 1

        # Allocate memory for numpy arrays
        self.malloc_rb(Nbasis, Npoints, Nquads, Nmodes=Nmodes, dtype=dtype)
        pass

    def seed(self, index_seed, trsp):
        """Seed the greedy algorithm.

        Seeds the first entries in the errors, indices, basis, and alpha arrays
        for use with the standard greedy algorithm for producing a reduced basis
        representation.

        Input
        -----
        index_seed     -- array index for seed point in training set
        trsp         -- the training space of functions

        Examples
        --------

        If rb is an instance of StandardRB, 0 is the array index associated
        with the seed, and T is the training set then do::

          >>> rb.seed(0, T)

        """

        self.errors[0] = np.max(self.norms) ** 2
        self.indices[0] = index_seed
        self.basis[0] = trsp[index_seed] / self.norms[index_seed]
        self.basisnorms[0] = self.norms[index_seed]
        self.alpha[0] = self.alpha_arr(self.basis[0], trsp)
        pass

    def iter(self, step, errs, trsp):
        """One iteration of standard reduced basis greedy algorithm.

        Updates the next entries of the errors, indices, basis, and
        alpha arrays.

        Input
        -----
        step -- current iteration step
        trsp -- the training space of functions

        Examples
        --------

        If rb is an instance of StandardRB and iter=13 is the 13th
        iteration of the greedy algorithm then the following code
        snippet generates the next (i.e., 14th) entry of the errors,
        indices, basis, and alpha arrays::

          >>> rb.iter(13)

        """

        self.errors[step + 1] = np.max(errs)
        self.indices[step + 1] = np.argmax(errs)
        self.basis[step + 1], self.basisnorms[step + 1] = self.add_basis(
            trsp[self.indices[step + 1]], self.basis[: step + 1]
        )
        self.alpha[step + 1] = self.alpha_arr(self.basis[step + 1], trsp)
        pass

    def make(self, index_seed, trsp, tol, verbose=True, timerQ=False):
        """Make a reduced basis using the standard greedy algorithm.

        Input
        -----
        index_seed     -- array index for seed point in training set
        trsp         -- the training space of functions
        tol            -- tolerance that terminates the greedy algorithm
        verbose        -- print projection errors to screen (default True)
        timer        -- print elapsed time (default False)

        Examples
        --------
        If rb is the StandardRB class instance, 0 the seed index, and
        T the training set then do::

          >>> rb.make(0, T, 1e-12)

        To prevent displaying any print to screen, set the `verbose`
        keyword argument to `False`::

          >>> rb.make(0, T, 1e-12, verbose=False)

        """

        # Seed the greedy algorithm
        self.seed(index_seed, trsp)
        flag = 0

        # The standard greedy algorithm with fixed training set
        if verbose:
            print("\nIter", "\t", "Error")
        if timerQ:
            t0 = time.time()
        for ctr in range(len(self.errors) - 1):
            if verbose:
                print(ctr + 1, "\t", self.errors[ctr] / self.errors[0])

            # Check if tolerance is met
            if self.errors[ctr] <= tol:
                flag = 1
                break

            # Single iteration and update errors, indices, basis, alpha arrays
            errs = self.proj_errors_from_alpha(self.alpha[: ctr + 1], norms=self.norms)
            self.iter(ctr, errs, trsp)

        if timerQ:
            print("\nElapsed time =", time.time() - t0)

        # Actual number of basis functions
        if flag == 1:
            self.size = ctr
        else:
            self.size = ctr + 2

        # Trim excess allocated entries
        self.trim(self.size)
        pass

    def projection(self, h):
        return self.rb_project(h, self.basis)

    def trim(self, num):
        """Trim zeros from remaining entries"""
        self.errors = self.errors[:num]
        self.indices = self.indices[:num]
        self.basis = self.basis[:num]
        self.alpha = self.alpha[:num]
        pass


##############################################
class StandardEIM(eim.EmpiricalInterpolation):
    def __init__(self, Nbasis, Nquads, Nmodes=1, dtype="complex"):
        eim.EmpiricalInterpolation.__init__(self)

        # Allocate memory for numpy arrays
        self.malloc_ei(Nbasis, Nquads, Nmodes=Nmodes, dtype=dtype)
        self.modes = Nmodes
        self.quads = Nquads

    def seed(self, e):
        """Seed the algorithm"""
        self.indices[0] = np.argmax(np.abs(e))
        self.R[0] = e
        self.invV[:1, :1] = self.next_invV_col(self.R[:1], self.indices[:1])
        pass

    def iter(self, step, e):
        """One iteration in the empirical interpolation greedy algorithm"""

        ctr = step + 1

        # Compute interpolant residual
        c = self.coefficient(self.invV[:ctr, :ctr], e, self.indices[:ctr])
        r = self.residual(e, c, self.R[:ctr])

        # Update
        self.indices[ctr] = np.argmax(np.abs(r))
        self.R[ctr] = r
        self.invV[:, ctr][: ctr + 1] = self.next_invV_col(
            self.R[: ctr + 1], self.indices[: ctr + 1]
        )
        pass

    def make(self, basis, verbose=True, timerQ=False):
        """Make an empirical interpolant using the standard greedy algorithm"""

        # Seed the greedy algorithm
        self.seed(basis[0])

        # EIM algorithm with reduced complexity for inverting the van der Monde matrix
        if verbose:
            print("\nIter", "\t", "Indices")
        if timerQ:
            t0 = time.time()
        dim = len(basis)
        for ctr in range(dim - 1):
            if verbose:
                print(ctr + 1, "\t", self.indices[ctr])

            # Single iteration
            self.iter(ctr, basis[ctr + 1])

        if timerQ:
            print("\nElapsed time =", time.time() - t0)

        # Compute interpolant matrix 'B'
        # if flag == 1:
        #     self.size = ctr
        # else:
        #     self.size = ctr+2
        # self.trim(ctr+1)
        self.trim(ctr + 2)
        self.make_interpolant()
        self.size = len(self.indices)
        pass

    def make_interpolant(self):
        self.B = self.eim_interpolant(self.invV, self.R)
        pass

    def interpolate(self, h):
        return self.eim_interpolate(h, self.indices, self.B)

    def trim(self, num):
        """Trim zeros from remaining entries"""
        if num > 0:
            self.indices = self.indices[:num]
            self.R = self.R[:num]
            self.invV = self.invV[:num, :num]
            self.B = self.B[:num, :]
        pass


##############################################
class StandardRBEIM(greedy.ReducedBasis, eim.EmpiricalInterpolation):
    def __init__(
        self,
        Nbasis,
        Npoints,
        Nquads,
        inner,
        inner_type,
        Nmodes=1,
        norms=None,
        dtype="complex",
    ):
        self.rb = StandardRB(
            Nbasis,
            Npoints,
            Nquads,
            inner,
            inner_type,
            Nmodes=Nmodes,
            norms=norms,
            dtype=dtype,
        )
        self.eim = StandardEIM(Nbasis, Nquads, dtype=dtype)
        greedy.ReducedBasis.__init__(self, inner, inner_type)
        eim.EmpiricalInterpolation.__init__(self)

    def seed(self, rb_index_seed, trsp):
        """Seed the algorithms"""
        self.rb.seed(rb_index_seed, trsp)
        self.eim.seed(self.rb.basis[0])
        pass

    def iter(self, step, trsp, e):
        """One iteration in the reduced basis and empirical
        interpolation greedy algorithms
        """
        self.rb.iter(step, trsp)
        self.eim.iter(step, e)

    def make(self, rb_index_seed, trsp, tol, verbose=True):
        """Make a reduced basis and an empirical interpolant simultaneously"""

        # Seed the greedy algorithm
        self.seed(rb_index_seed, trsp)

        # Greedy algorithm
        if verbose:
            print("\nIter", "\t", "Error", "\t", "Indices")
        dim = len(self.rb.basis[:, 0])
        for ctr in range(dim - 1):
            if verbose:
                print(ctr + 1, "\t", self.rb.errors[ctr], "\t", self.eim.indices[ctr])

            # Check if tolerance is met
            if self.rb.errors[ctr] <= tol:
                break

            # Single iteration
            self.iter(ctr, trsp, self.rb.basis[ctr + 1])

        # Compute interpolant matrix 'B'
        self.rb.trim(ctr)
        self.eim.trim(ctr)
        self.eim.make_interpolant()
        self.size = ctr
        pass
