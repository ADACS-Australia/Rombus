# --- integrals.py ---

"""
    Classes and functions for computing inner products of functions
"""

import numpy as np
from .training import TrainingTools


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def rate_to_num(a, b, rate):
    """Convert sample rate to sample numbers in [a,b]"""
    return np.floor(float(b - a) * rate) + 1


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def num_to_rate(a, b, num):
    """Convert sample numbers in [a,b] to sample rate"""
    return (num - 1.0) / np.float(b - a)


##############################################
class Quadratures(TrainingTools):

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __init__(self):
        TrainingTools.__init__(self)
        self.switch_dict = {
            "riemann-rate": self.riemann_rate,
            "riemann-num": self.riemann_num,
            "trapezoidal-rate": self.trapezoidal_rate,
            "trapezoidal-num": self.trapezoidal_num,
            "chebyshev": self.chebyshev,
            "chebyshev-gauss-lobatto": self.chebyshev_gauss_lobatto,
            "legendre": self.legendre,
            "legendre-gauss-lobatto": self.legendre_gauss_lobatto,
            #'roq': self.roq
        }
        self.options = self.switch_dict.keys()

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __call__(self, args, quad_type):  # , multidomainQ=False):

        dim = np.shape(args[:-1])
        num_or_rate = args[-1]

        # Check that the number of domains equals to the number of samples/sample rates specified
        if type(num_or_rate) == list:
            if len(num_or_rate) != dim[0] - 1:
                raise Exception(
                    "Expecting equal number of intervals and number of samples/sample rates."
                )

        # Check how many quadrature rules are specified
        if type(quad_type) == list:
            type_quad_type = "list"
            if len(quad_type) != len(num_or_rate):
                raise Exception(
                    "Expecting equal number of intervals and specified quadrature rules."
                )
        else:
            type_quad_type = "str"

        # Make quadrature rules for single or multiple domains
        if len(dim) == 1:

            # Single integration domain
            if dim[0] == 2:
                ans = self.switch_dict[quad_type](*args)

            # Multiple integration domains
            if dim[0] > 2:
                if type_quad_type == "str":
                    ans = np.hstack(
                        [
                            self.switch_dict[quad_type](
                                args[ii], args[ii + 1], num_or_rate[ii]
                            )
                            for ii in range(dim[0] - 1)
                        ]
                    )
                elif type_quad_type == "list":
                    ans = np.hstack(
                        [
                            self.switch_dict[quad_type[ii]](
                                args[ii], args[ii + 1], num_or_rate[ii]
                            )
                            for ii in range(dim[0] - 1)
                        ]
                    )

        elif len(dim) > 1:
            # CHECKME
            # For integrating in multiple dimensions
            nodes, weights = [], []
            for ii in range(len(args[0])):
                arg = np.transpose(args)[ii]
                quad = quad_type[ii]
                nws = self.switch_dict[quad](*arg)
                nodes.append(nws[0])
                weights.append(nws[1])
            weights_nd = np.transpose(self.tuple_to_vstack(self.meshgrid2(*weights)))
            ans = [
                self.tuple_to_vstack(self.meshgrid2(*nodes)).T,
                np.array(
                    [np.prod(weights_nd[ii]) for ii in range(len(weights_nd[:, 0]))]
                ),
            ]

        return ans

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def riemann_num(self, a, b, n):
        """
        Uniformly sampled array using Riemann quadrature rule
        over given interval with given number of samples

        Input
        -----
        a -- start of interval
        b -- end of interval
        n -- number of quadrature points

        Output
        ------
        nodes   -- quadrature nodes
        weights -- quadrature weights
        """
        nodes = np.linspace(a, b, num=n)
        weights = np.ones(n, dtype="double")
        return [nodes, (b - a) / (n - 1.0) * weights]

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def riemann_rate(self, a, b, rate):
        """
        Uniformly sampled array using Riemann quadrature rule
        over given interval with given sample rate

        Input
        -----
        a    -- start of interval
        b    -- end of interval
        rate -- sample rate

        Output
        ------
        nodes   -- quadrature nodes
        weights -- quadrature weights
        """
        n = rate_to_num(a, b, rate)
        return self.riemann_num(a, b, n)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def trapezoidal_num(self, a, b, n):
        """
        Uniformly sampled array using trapezoidal quadrature rule
        over given interval with given number of samples

        Input
        -----
        a -- start of interval
        b -- end of interval
        n -- number of quadrature samples

        Output
        ------
        nodes   -- quadrature nodes
        weights -- quadrature weights
        """
        nodes = np.linspace(a, b, num=n)
        weights = np.ones(n, dtype="double")
        weights[0] /= 2.0
        weights[-1] /= 2.0
        return [nodes, weights * (b - a) / (n - 1.0)]

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def trapezoidal_rate(self, a, b, rate):
        """
        Uniformly sampled array using trapezoidal quadrature rule
        over given interval with given sample rate

        Input
        -----
        a    -- start of interval
        b    -- end of interval
        rate -- sample rate

        Output
        ------
        nodes   -- quadrature nodes
        weights -- quadrature weights
        """
        n = rate_to_num(a, b, rate)
        return self.trapezoidal_num(a, b, n)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def chebyshev(self, a, b, n):
        """
        Uniformly sampled array using Chebyshev-Gauss quadrature rule
        over given interval with given number of samples

        Input
        -----
        a -- start of interval
        b -- end of interval
        n -- number of quadrature samples

        Output
        ------
        nodes   -- quadrature nodes
        weights -- quadrature weights
        """
        num = int(n) - 1.0
        nodes = np.array(
            [
                -np.cos(np.pi * (2.0 * ii + 1.0) / (2.0 * num + 2.0))
                for ii in range(int(n))
            ]
        )
        weights = np.pi / (num + 1.0) * np.sqrt(1.0 - nodes**2)
        return [nodes * (b - a) / 2.0 + (b + a) / 2.0, weights * (b - a) / 2.0]

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def chebyshev_gauss_lobatto(self, a, b, n):
        """
        Uniformly sampled array using Chebyshev-Gauss-Lobatto quadrature rule
        over given interval with given number of samples

        Input
        -----
        a -- start of interval
        b -- end of interval
        n -- number of quadrature samples

        Output
        ------
        nodes   -- quadrature nodes
        weights -- quadrature weights
        """
        num = int(n) - 1.0
        nodes = np.array([-np.cos(np.pi * ii / num) for ii in range(int(n))])
        weights = np.pi / num * np.sqrt(1.0 - nodes**2.0)
        weights[0] /= 2.0
        weights[-1] /= 2.0
        return [nodes * (b - a) / 2.0 + (b + a) / 2.0, weights * (b - a) / 2.0]

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def legendre(self, a, b, num):
        raise Exception("Legendre Gauss quadrature rule is not yet implemented.")
        pass

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def legendre_gauss_lobatto(self, a, b, num):
        raise Exception(
            "Legendre Gauss-Lobatto quadrature rule is not yet implemented."
        )
        pass


# 	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 	def roq(self, x, w, eim_indices, B):
# 		"""Reduced-order quadrature rule
#
# 		Input
# 		-----
# 		x 			-- original quadrature nodes (e.g., Chebyshev)
# 		w 			-- original quadrature weights
# 		eim_indices	-- array indices of empirical interpolation nodes
# 		B 			-- empirical interpolant operator, `B`
#
# 		Returns
# 		-------
# 		nodes 	-- reduced-order quadrature nodes
# 		weights -- reduced-order quadrature weights
#
# 		Examples
# 		--------
# 		Let x, w be 20 Chebyshev Gauss nodes and weights over the interval
# 		[10, 100] from
#
# 		  >>> qu = rp.Quadratures()
#
# 		  >>> x, w = qu([10, 100, 20], 'chebyshev')
#
# 		If `B` is the empirical interpolant operator so that the empirical
# 		interpolant acting on a function `h` is I[h] = \sum_{i=1}^m B_i*h_i
# 		and `inds` are the array indices associated with the empirical
# 		interpolation nodes then
#
# 		  >>> roq_x, roq_w = rp.Quadratures()(x, w, inds, B)
#
# 		gives the reduced-order quadrature nodes (`roq_x`) and weights
# 		(`roq_w`).
#
# 		"""
#
# 		nodes = x[eim_indices]
# 		weights = np.dot(B, w)
# 		return [nodes, weights]


##############################################
class Real:

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __init__(self, args, quad_type="trapezoidal-num"):
        self.quad_type = quad_type
        self.nodes, self.weights = Quadratures()(args, quad_type)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def integral(self, h):
        """Integral of a function"""
        return np.dot(self.weights, np.real(h))

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def dot(self, h1, h2):
        """Dot product of two functions"""
        dim = np.shape(h1)
        h1real = np.real(h1)
        h2real = np.real(h2)
        if len(dim) == 1:
            return np.dot(self.weights, h1real * h2real)
        else:
            return np.dot(
                self.weights, np.sum(h1real[ii] * h2real[ii] for ii in range(dim[0]))
            )

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def norm(self, h):
        """Norm of function"""
        hreal = np.real(h)
        return np.sqrt(np.real(self.dot(hreal, hreal)))

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def normalize(self, h):
        """Normalize a function"""
        return h / self.norm(h)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def match(self, h1, h2):
        """Match integral"""
        h1n = self.normalize(h1)
        h2n = self.normalize(h2)
        return np.real(self.dot(h1n, h2n))

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def mismatch(self, h1, h2):
        """Mismatch integral (1-match)"""
        return 1.0 - self.match(h1, h2)


##############################################
class Complex:

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __init__(self, args, quad_type="trapezoidal-num"):
        self.quad_type = quad_type
        self.nodes, self.weights = Quadratures()(args, quad_type)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def integral(self, h):
        """Integral of a function"""
        return np.dot(self.weights, h)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def dot(self, h1, h2):
        """Dot product of two functions"""
        dim = np.shape(h1)
        if len(dim) == 1:
            return self.integral(np.conj(h1) * h2)
        else:
            return self.integral(
                np.sum(np.conj(h1[ii]) * h2[ii] for ii in range(dim[0]))
            )

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def norm(self, h):
        """Norm of function"""
        return np.sqrt(np.real(self.dot(h, h)))

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def normalize(self, h):
        """Normalize a function"""
        return h / self.norm(h)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def match(self, h1, h2):
        """Match integral"""
        h1n = self.normalize(h1)
        h2n = self.normalize(h2)
        return np.real(self.dot(h1n, h2n))

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def mismatch(self, h1, h2):
        """Mismatch integral (1-match)"""
        return 1.0 - self.match(h1, h2)


##############################################
class InnerProduct(Real, Complex):
    """Integrals for computing inner products and norms of functions"""

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __init__(self, args, quad_type="trapezoidal-num"):

        Real.__init__(self, args, quad_type)
        self.real = Real(args, quad_type)

        Complex.__init__(self, args, quad_type)
        self.complex = Complex(args, quad_type)

        self.switch_dict = {
            "complex": self.complex,
            "real": self.real,
        }
        self.options = self.switch_dict.keys()

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __call__(self, h1, h2, inner_type):
        return self.dot(h1, h2, inner_type)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def integral(self, h, inner_type):
        """Integral of a function"""
        return self.switch_dict[inner_type].integral(h)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def dot(self, h1, h2, inner_type):
        """Dot product of two functions"""
        return self.switch_dict[inner_type].dot(h1, h2)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def norm(self, h, inner_type):
        """Norm of function"""
        return np.sqrt(np.real(self.dot(h, h, inner_type)))

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def normalize(self, h, inner_type):
        """Normalize a function"""
        return h / self.norm(h, inner_type)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def match(self, h1, h2, inner_type):
        """Match integral"""
        return self.switch_dict[inner_type].match(h1, h2)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def mismatch(self, h1, h2, inner_type):
        """Mismatch integral (1-match)"""
        return self.switch_dict[inner_type].mismatch(h1, h2)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def Linfty(self, h):
        """L-infinity norm"""
        # FIXME: h cannot be a multi-dim array
        return np.max(np.abs(h))

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def Ln(self, h, n):
        """L-n norm"""
        # FIXME: h cannot be a multi-dim array
        if n > 0:
            return (np.dot(self.weights, np.abs(h) ** n)) ** (1.0 / n)
        pass

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def L2(self, h):
        """L-2 norm"""
        # FIXME: h cannot be a multi-dim array
        return self.Ln(h, 2)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def _test_monomial(self, n=0):
        """Test integration rule by integrating the monomial x**n"""
        ans = self.integral(self.nodes**n, "real")
        a, b = self.nodes[0], self.nodes[-1]
        expd = (b ** (n + 1) - a ** (n + 1)) / (n + 1)

        print("\nExpected value for integral =", expd)
        print("Computed value for integral =", ans)
        print("\nAbsolute difference =", expd - ans)
        print("Relative difference =", 1.0 - ans / expd)

        pass
