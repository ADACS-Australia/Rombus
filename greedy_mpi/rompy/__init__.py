# --- __init__.py ---

"""RomPy
   =====
   
   Provides
     1. Functionality to build problem-specific greedy algorithms
     2. Reduced basis representations of function spaces
     3. Empirical interpolation methods
     4. Reduced-order quadrature rules
     5. Empirical Fourier transforms
   
   How to use the documentation
   -----------------------------
   Documentation is available via docstrings provided with the code.
   
   We recommend exploring the docstrings using 
   `IPython <http://ipython.scipy.org>`, an advanced Python shell with 
   TAB-completion and introspection capabilities. See below for further 
   instructions.
   
   The docstring examples assume that `rompy` has been imported as `rp`::
   
     >>> import rompy as rp
	
   Code snippets are indicated by three greater-than signs.
   
   Use the built-in ``help`` function to view a function's or class's 
   docstring::
   
     >>> help(rp.greedy)
   
   Available subpackages
   ---------------------
   algorithms
       Standard reduced-order modeling routines
   
   eim
       Core empirical interpolation tools
       
   greedy
       Core greedy algorithm tools
   
   integrals
       Tools for quadrature rules and integration
       
   lib
       Basic functions used by several subpackages
   
   training
       Core training set generation tools
   
   Viewing documentation using IPython
   -----------------------------------
   Start IPython and load RomPy (``import rompy as rp``), which 
   will import `rompy` under the alias `rp`. Then, use the ``cpaste`` 
   command to paste examples into the shell. To see which functions 
   are available in `rompy`, type ``rp.<TAB>`` (where ``<TAB>`` refers 
   to the TAB key), or use ``rp.greedy?<ENTER>`` (where ``<ENTER>``
   refers to the ENTER key) to narrow down the list. To view the 
   docstring for a function or class, use ``rp.greedy?<ENTER>`` (to 
   view the docstring) and ``rp.greedy??<ENTER>`` (to view the source 
   code).
   """

__copyright__ = "Copyright (C) 2013 Chad R. Galley"
__email__ = "crgalley@tapir.caltech.edu, crgalley@gmail.com"
__author__ = "Chad Galley"


import numpy as np
import time
import random


from integrals import *
from greedy import *
from eim import *
from lib import *
from fit import *


import algorithms
import training
#import eft
