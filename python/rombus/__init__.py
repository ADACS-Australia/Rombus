"""

   How to use the documentation
   -----------------------------
   Documentation is available via docstrings provided with the code.

   We recommend exploring the docstrings using
   `IPython <http://ipython.scipy.org>`, an advanced Python shell with
   TAB-completion and introspection capabilities. See below for further
   instructions.

   Code snippets are indicated by three greater-than signs.

   Use the built-in ``help`` function to view a function's or class's
   docstring::

     >>> help(rb.model)

   Viewing documentation using IPython
   -----------------------------------
   Start IPython and load Rombus (``import rombus as rb``), which
   will import `rombus` under the alias `rb`. Then, use the ``cpaste``
   command to paste examples into the shell. To see which functions
   are available in `rombus`, type ``rb.<TAB>`` (where ``<TAB>`` refers
   to the TAB key), or use ``rb.model?<ENTER>`` (where ``<ENTER>``
   refers to the ENTER key) to narrow down the list. To view the
   docstring for a function or class, use ``rb.model?<ENTER>`` (to
   view the docstring) and ``rb.model??<ENTER>`` (to view the source
   code).
   """
from importlib import metadata # make sure to import metadata explicitly
__version__ = metadata.version(__package__ or __name__)
