# --- greedy.py ---

"""
    Classes for building reduced basis greedy algorithms
"""

__author__ = "Chad Galley <crgalley@gmail.com>"

import numpy as np

# from __init__ import np
# import adipy
from . import lib


##############################################
class IteratedModifiedGramSchmidt:
    """Iterated modified Gram-Schmidt algorithm for building an orthonormal basis.
    Algorithm from Hoffman, `Iterative Algorithms for Gram-Schmidt Orthogonalization`.
    """

    def __init__(self, inner, inner_type):
        self.inner = inner
        self.inner_type = inner_type

    def add_basis(self, h, basis, a=0.5, max_iter=3):
        """Given a function, h, find the corresponding basis
        function orthonormal to all previous ones
        """
        norm = self.inner.norm(h, self.inner_type)
        e = h / norm
        flag, ctr = 0, 1
        while flag == 0:
            for b in basis:
                e -= b * self.inner.dot(b, e, self.inner_type)
            new_norm = self.inner.norm(e, self.inner_type)

            # Iterate, if necessary
            if new_norm / norm <= a:
                norm = new_norm
                ctr += 1
                if ctr > max_iter:
                    print(
                        (
                            f"Max number of iterations ({str(max_iter)}) "
                            "reached in iterated Gram-Schmidt. Basis "
                            "may not be orthonormal."
                        )
                    )
            else:
                flag = 1

        return [e / new_norm, new_norm]

    def make_basis(self, hs, norms=False, a=0.5, max_iter=3):
        """Given a set of functions, hs, find the corresponding
        orthonormal set of basis functions.
        """

        dim = np.shape(hs)
        basis = np.zeros(dim, dtype=hs.dtype)
        basis[0] = self.inner.normalize(hs[0], self.inner_type)
        if norms:
            norm = np.zeros(dim[0], dtype="double")
            norm[0] = self.inner.norm(hs[0], self.inner_type)

        for ii in range(1, dim[0]):
            if norms:
                basis[ii], norm[ii] = self.add_basis(
                    hs[ii], basis[:ii], a=a, max_iter=max_iter
                )
            else:
                basis[ii], _ = self.add_basis(
                    hs[ii], basis[:ii], a=a, max_iter=max_iter
                )

        if norms:
            return [np.array(basis), norm]
        else:
            return np.array(basis)


##############################################
# class ReducedSpline1d:
#
#     def __init__(self):
#         pass
#
#     def autodiff(self, x, fn, n=4, args=None):
#         """ x ~ f
#             f ~ gwphase_3point5PN
#             n ~ 4 (spline degree + 1)
#             d = d/dx ~ d/df
#         """
#         ad = adipy.adn(x, n)
#         p = lib.fneval(ad, fn, args=args)
#         return p.d(n)
#
#     def delta(self, tol, deriv_n, n):
#         C3 = 5./384.
#         return (tol/C3/np.abs(deriv_n))**(1./n)


##############################################
class ReducedBasis:
    def __init__(self, inner, inner_type):
        self.inner = inner
        self.inner_type = inner_type

    def malloc_rb(self, Nbasis, Npoints, Nquads, Nmodes=1, dtype="complex"):
        """Allocate memory for numpy arrays used for making reduced basis"""
        self.errors = lib.malloc("double", Nbasis)
        self.indices = lib.malloc("int", Nbasis)
        if Nmodes == 1:
            self.basis = lib.malloc(dtype, Nbasis, Nquads)
        elif Nmodes > 1:
            self.basis = lib.malloc(dtype, Nbasis, Nmodes, Nquads)
        else:
            raise Exception("Expected positive number of modes.")
        self.basisnorms = lib.malloc("double", Nbasis)
        self.alpha = lib.malloc(dtype, Nbasis, Npoints)

    def alpha1(self, e, h):
        """Inner product of a basis function e with a function h:
        alpha(e,h) = <e, h>
        """
        return self.inner.dot(e, h, self.inner_type)

    def alpha_arr(self, e, hs):
        """Inner products of a basis function e with an array of functions hs"""
        return np.array([self.alpha1(e, hh) for hh in hs])

    def proj_error_from_basis(self, basis, h):
        """Square of the projection error of a function h on basis"""
        norm = np.real(self.inner.norm(h, self.inner_type))
        dim = len(basis[:, 0])
        return norm**2 - np.sum(
            np.abs(self.alpha1(basis[ii], h)) ** 2 for ii in range(dim)
        )

    def proj_errors_from_basis(self, basis, hs):
        """Square of the projection error of functions hs on basis"""
        # norms = np.real([self.inner.norm(hh, self.inner_type) for hh in hs])
        # dim = len(basis[:,0])
        # return [norms**2-np.sum(np.abs(self.alpha1(basis[ii], hh))**2
        #         for ii in range(dim)) for hh in hs]
        return [self.proj_error_from_basis(basis, hh) for hh in hs]

    def proj_mismatch_from_basis(self, basis, h):
        """Mismatch of a function h with its projection onto the basis"""
        norms = np.real(self.inner.norm(h, self.inner_type))
        dim = len(basis[:, 0])
        return (
            1.0
            - np.real(
                np.sum(np.abs(self.alpha1(basis[ii], h)) ** 2 for ii in range(dim))
            )
            / norms
        )

    def proj_errors_from_alpha(self, alpha, norms=None):
        """Square of the projection error of a function h on basis
        in terms of pre-computed alpha matrix.
        """
        if norms is None:
            norms = np.ones(len(alpha[0]), dtype="double")
        else:
            norms = norms
        dim = len(alpha[:, 0])
        return norms**2 - np.sum(np.abs(alpha[ii]) ** 2 for ii in range(dim))

    def projection_from_basis(self, h, basis):
        """Project a function h onto the basis functions"""
        # return np.array([ee*self.alpha1(ee, h) for ee in basis])
        return np.sum(ee * self.alpha1(ee, h) for ee in basis)

    def projection_from_alpha(self, basis, alpha):
        """Project a function h onto the basis functions in terms of
        pre-computed alpha matrix entry.
        """
        return np.array([np.sum(basis[ii] * alpha[ii]) for ii in range(len(basis))])
