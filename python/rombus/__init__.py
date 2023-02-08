# --- __init__.py ---

"""Rombus
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

   The docstring examples assume that `rombus` has been imported as `rb`::

     >>> import rombus as rb

   Code snippets are indicated by three greater-than signs.

   Use the built-in ``help`` function to view a function's or class's
   docstring::

     >>> help(rb.greedy)

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
   Start IPython and load Rombus (``import rombus as rb``), which
   will import `rombus` under the alias `rb`. Then, use the ``cpaste``
   command to paste examples into the shell. To see which functions
   are available in `rombus`, type ``rb.<TAB>`` (where ``<TAB>`` refers
   to the TAB key), or use ``rb.greedy?<ENTER>`` (where ``<ENTER>``
   refers to the ENTER key) to narrow down the list. To view the
   docstring for a function or class, use ``rb.greedy?<ENTER>`` (to
   view the docstring) and ``rb.greedy??<ENTER>`` (to view the source
   code).
   """
