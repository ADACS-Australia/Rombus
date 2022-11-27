# --- fit.py ---

"""
	Classes and functions for fitting functions at the greedy-selected parameter values.
"""


import numpy as np
from scipy.interpolate import splrep
from scipy.interpolate import splev

# from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import scipy.optimize as opt
from . import eim, lib


##############################################
# The following three functions (_general_function, _weighted_general_function,
# curve_fit) are from scipy.optimize. I have changed the error handling here
# so that curve_fit returns the parameters found, even if they are not optimal.
# The original code raises an exception and aborts the program. Using a `try` and `except`
# pair does not seem to retain the (suboptimal) fit parameters.


def _general_function(params, xdata, ydata, function):
    return function(xdata, *params) - ydata


def _weighted_general_function(params, xdata, ydata, function, weights):
    return weights * (function(xdata, *params) - ydata)


def curve_fit(f, xdata, ydata, p0=None, sigma=None, **kw):
    """
    Use non-linear least squares to fit a function, f, to data.

    Assumes ``ydata = f(xdata, *params) + eps``

    Parameters
    ----------
    f : callable
        The model function, f(x, ...).  It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.
    xdata : An N-length sequence or an (k,N)-shaped array
        for functions with k predictors.
        The independent variable where the data is measured.
    ydata : N-length sequence
        The dependent data --- nominally f(xdata, ...)
    p0 : None, scalar, or M-length sequence
        Initial guess for the parameters.  If None, then the initial
        values will all be 1 (if the number of parameters for the function
        can be determined using introspection, otherwise a ValueError
        is raised).
    sigma : None or N-length sequence
        If not None, this vector will be used as relative weights in the
        least-squares problem.

    Returns
    -------
    popt : array
        Optimal values for the parameters so that the sum of the squared error
        of ``f(xdata, *popt) - ydata`` is minimized
    pcov : 2d array
        The estimated covariance of popt.  The diagonals provide the variance
        of the parameter estimate.

    See Also
    --------
    leastsq

    Notes
    -----
    The algorithm uses the Levenberg-Marquardt algorithm through `leastsq`.
    Additional keyword arguments are passed directly to that algorithm.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import curve_fit
    >>> def func(x, a, b, c):
    ...     return a*np.exp(-b*x) + c

    >>> x = np.linspace(0,4,50)
    >>> y = func(x, 2.5, 1.3, 0.5)
    >>> yn = y + 0.2*np.random.normal(size=len(x))

    >>> popt, pcov = curve_fit(func, x, yn)

    """
    if p0 is None:
        # determine number of parameters by inspecting the function
        import inspect

        args, varargs, varkw, defaults = inspect.getargspec(f)
        if len(args) < 2:
            msg = "Unable to determine number of fit parameters."
            raise ValueError(msg)
        if "self" in args:
            p0 = [1.0] * (len(args) - 2)
        else:
            p0 = [1.0] * (len(args) - 1)

    if np.isscalar(p0):
        p0 = array([p0])

    args = (xdata, ydata, f)
    if sigma is None:
        func = _general_function
    else:
        func = _weighted_general_function
        args += (1.0 / np.asarray(sigma),)

    # Remove full_output from kw, otherwise we're passing it in twice.
    return_full = kw.pop("full_output", False)
    res = leastsq(func, p0, args=args, full_output=1, **kw)
    (popt, pcov, infodict, errmsg, ier) = res

    if ier not in [1, 2, 3, 4]:
        msg = "Optimal parameters not found: " + errmsg
        # raise RuntimeError(msg)
        print(msg)
        return popt, pcov
        # if return_full:
        #    return popt, pcov, infodict, errmsg, ier
        # else:
        #    return popt, pcov

    if (len(ydata) > len(p0)) and pcov is not None:
        s_sq = (func(popt, *args) ** 2).sum() / (len(ydata) - len(p0))
        pcov = pcov * s_sq
    else:
        pcov = np.inf

    if return_full:
        return popt, pcov, infodict, errmsg, ier
    else:
        return popt, pcov


##############################################
class Spline1d:
    """Class for building one dimensional spline interpolants"""

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __init__(self, x, y, k=3):
        """Initialize Spline1d class. Builds 1d spline interpolant
        of degree k from x and y data.

        Input
        -----
           x   -- given array of parameter values
           y   -- function values to be interpolated (e.g., at empirical
                           interpolation nodes
           k    -- degree of spline (default is 3)
        """

        args = np.argsort(x)  # `x` values must be sorted in ascending order
        self.dim = np.shape(y)
        self.degree = k
        if len(self.dim) == 1:
            self.knots, self.coeffs, _ = splrep(x[args], y[args], k=k)
        elif len(self.dim) > 1:
            self.coeffs = []
            for yy in y:
                temp = splrep(x[args], yy[args], k=k)
                self.coeffs.append(temp[1])
                self.knots = temp[0]
        else:
            raise Exception("Expected a non-empty array.")

        self.fitparams = [self.knots, self.coeffs, self.degree]

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __call__(self, p, der=0):
        return self.fit(p, der)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def fit(self, p, der=0):
        if len(self.dim) == 1:
            return splev(p, self.fitparams, der)
        elif len(self.dim) > 1:
            return np.array(
                [
                    splev(p, [self.knots, self.coeffs[ii], self.degree])
                    for ii in range(self.dim[0])
                ]
            )


##############################################
class Polynomial:

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __init__(self, x, y, deg=3):

        # Map x to reference interval, [-1,1]
        ref = lib.map_intervals(x, -1.0, 1.0)

        self.shape = np.shape(y)
        if len(self.shape) == 1:
            self.fitparams_ref = np.polyfit(ref, y, deg=deg)
            self.fitparams = self.map_poly_coeffs(
                self.fitparams_ref, ref, np.min(x), np.max(x)
            )
        elif len(self.shape) > 1:
            if len(np.shape(deg)) == 0:
                self.fitparams_ref = [
                    np.polyfit(ref, y[ii], deg=deg) for ii in range(self.shape[0])
                ]
            else:
                self.fitparams_ref = [
                    np.polyfit(ref, y[ii], deg=deg[ii]) for ii in range(self.shape[0])
                ]
            self.fitparams = [
                self.map_poly_coeffs(pp, ref, np.min(x), np.max(x))
                for pp in self.fitparams_ref
            ]
        else:
            raise Exception("Expected a non-empty array.")
        pass

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def map_poly_coeffs(self, coeffs, x, a, b):
        """Map polynomial coefficients from p(x) to that on the interval [a,b]"""

        # Linear transformation from x to [a,b]
        M = (np.max(x) - np.min(x)) / (b - a)
        B = np.min(x) - M * a

        # Map the polynomial coefficients
        deg, mapped_coeffs = len(coeffs), []
        c = coeffs[::-1]  # Reverse order of polynomial coefficients
        for n in range(deg):
            mapped_coeffs.append(
                M**n
                * np.sum(
                    lib.choose(kk, n) * c[kk] * B ** (kk - n) for kk in range(n, deg)
                )
            )

        return mapped_coeffs[
            ::-1
        ]  # Reverse order of polynomial coefficients for use with np.polyval

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __call__(self, p):
        return self.fit(p)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def fit(self, p):
        if len(self.shape) == 1:
            return np.polyval(self.fitparams, p)
        elif len(self.shape) > 1:
            return np.array(
                [np.polyval(self.fitparams[ii], p) for ii in range(self.shape[0])]
            )


##############################################
class UserDefined:

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __init__(self, x, y, fn, p0=None, maxfev=1000):
        self.shape = y.shape
        self.fn = fn
        if len(self.shape) == 1:
            self.fitparams = curve_fit(fn, x, y, p0=p0, maxfev=maxfev)[0]
        elif len(self.shape) > 1:
            self.fitparams = [
                curve_fit(fn, x, y[ii], p0=p0, maxfev=maxfev)[0]
                for ii in range(self.shape[0])
            ]
        else:
            raise Exception("Expected a non-empty array.")
        pass

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __call__(self, p):
        return self.fit(p)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def fit(self, p):
        if len(self.shape) == 1:
            return self.fn(p, *self.fitparams)
        elif len(self.shape) > 1:
            return np.array(
                [self.fn(p, *self.fitparams[ii]) for ii in range(self.shape[0])]
            )


##############################################
# class Surrogate(Spline1d, Polynomial, UserDefined, eim.EmpiricalInterpolation):
class Surrogate(Spline1d, Polynomial, UserDefined):

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __init__(self, args, fit_type):
        # eim.EmpiricalInterpolation.__init__(self)

        if fit_type == "spline-1d":
            Spline1d.__init__(self, *args)
            self.fit = Spline1d(*args).fit
        elif fit_type == "poly":
            Polynomial.__init__(self, *args)
            self.fit = Polynomial(*args).fit
        elif fit_type == "userdef":
            UserDefined.__init__(self, *args)
            self.fit = UserDefined(*args).fit

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __call__(self, p, B):
        return eval(p, B)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def eval(self, p, B):
        return np.dot(self.fit(p), B)
