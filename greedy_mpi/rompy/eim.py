# --- eim.py ---

"""
	Classes for the empirical interpolation method
"""

__author__ = "Chad Galley <crgalley@gmail.com>"
import numpy as np

#from __init__ import np
from scipy.linalg.lapack import get_lapack_funcs
#from scipy.linalg import solve_triangular
import lib


##############################################
class LinAlg:
	"""Linear algebra functions needed for empirical interpolation class"""
	
	def __init__(self):
		pass
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# This is scipy.linalg's source code. For some reason my scipy doesn't recognize
	# the check_finite option, which may help with speeding up. Because this may be an 
	# issue related to the latest scipy.linalg version, I'm reproducing that code here
	# to guarantee future compatibility.
	def solve_triangular(self, a, b, trans=0, lower=False, unit_diagonal=False, \
	                     overwrite_b=False, debug=False, check_finite=True):
	    """
	    Solve the equation `a x = b` for `x`, assuming a is a triangular matrix.
	
	    Parameters
	    ----------
	    a : (M, M) array_like
	        A triangular matrix
	    b : (M,) or (M, N) array_like
	        Right-hand side matrix in `a x = b`
	    lower : boolean
	        Use only data contained in the lower triangle of `a`.
	        Default is to use upper triangle.
	    trans : {0, 1, 2, 'N', 'T', 'C'}, optional
	        Type of system to solve:
	
	        ========  =========
	        trans     system
	        ========  =========
	        0 or 'N'  a x  = b
	        1 or 'T'  a^T x = b
	        2 or 'C'  a^H x = b
	        ========  =========
	    unit_diagonal : bool, optional
	        If True, diagonal elements of `a` are assumed to be 1 and
	        will not be referenced.
	    overwrite_b : bool, optional
	        Allow overwriting data in `b` (may enhance performance)
	    check_finite : bool, optional
	        Whether to check that the input matrices contain only finite numbers.
	        Disabling may give a performance gain, but may result in problems
	        (crashes, non-termination) if the inputs do contain infinities or NaNs.
	
	    Returns
	    -------
	    x : (M,) or (M, N) ndarray
	        Solution to the system `a x = b`.  Shape of return matches `b`.
	
	    Raises
	    ------
	    Exception
	        If `a` is singular
	
	    Notes
	    -----
	    .. versionadded:: 0.9.0
	
	    """
	
	    if check_finite:
	        a1, b1 = map(np.asarray_chkfinite,(a,b))
	    else:
	        a1, b1 = map(np.asarray, (a,b))
	    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
	        raise ValueError('expected square matrix')
	    if a1.shape[0] != b1.shape[0]:
	        raise ValueError('incompatible dimensions')
	    overwrite_b = False #overwrite_b or _datacopied(b1, b)
	    if debug:
	        print('solve:overwrite_b=',overwrite_b)
	    trans = {'N': 0, 'T': 1, 'C': 2}.get(trans, trans)
	    trtrs, = get_lapack_funcs(('trtrs',), (a1,b1))
	    x, info = trtrs(a1, b1, overwrite_b=overwrite_b, lower=lower, trans=trans, unitdiag=unit_diagonal)
	
	    if info == 0:
	        return x
	    if info > 0:
	        raise Exception("singular matrix: resolution failed at diagonal %s" % (info-1))
	    raise ValueError('illegal value in %d-th argument of internal trtrs' % -info)
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def transpose(self, a):
		dim = a.shape
		if len(dim) != 2:
			raise ValueError('Expected a matrix')
		aT = np.zeros(dim[::-1], dtype=a.dtype)
		for ii, aa in enumerate(a):
			aT[:,ii] = a[ii]
		return aT
	
	
##############################################
class EmpiricalInterpolation(LinAlg):
	"""
	Class for building an empirical interpolant
	"""
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def __init__(self):
		LinAlg.__init__(self)
		
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def malloc(self, Nbasis, Nquads, Nmodes=1, dtype='complex'):
		self.indices = lib.malloc('int', Nbasis) 
		self.invV = lib.malloc(dtype, Nbasis, Nbasis) 
		self.R = lib.malloc(dtype, Nbasis, Nquads)
		self.B = lib.malloc(dtype, Nbasis, Nquads)
		pass
			
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def coefficient(self, invV, e, indices):
		return np.dot(invV.T, e[indices])
		
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def residual(self, e, c, R):
		"""Difference between a basis function 'e' and its empirical interpolation"""
		return e - np.dot(c, R)
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def next_invV_col(self, R, indices, check_finite=False):
		b = np.zeros(len(indices), dtype=R.dtype)
		b[-1] = 1.
		return self.solve_triangular(R[:,indices], b, lower=False, check_finite=check_finite)
		#return solve_triangular(R[:,indices], b, lower=False, check_finite=check_finite)
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def eim_interpolant(self, invV, R):
		"""The empirical interpolation matrix 'B'"""
		return np.dot(invV, R)
	
	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	def eim_interpolate(self, h, indices, B):
		"""Empirically interpolate a function"""
		dim = np.shape(h)
		if len(dim) == 1:
			return np.dot(h[indices], B)
		elif len(dim) > 1:
			return np.array([np.dot(h[ii][indices], B) for ii in range(dim[0])])
	
