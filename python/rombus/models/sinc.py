import numpy as np
from rombus.model import RombusModel
from typing import NamedTuple
import math

# This sample code provides a starting point for building your own Reduced Order Models
# Edit it to suit your needs.  In its initial form, it provides a model for building
# a reduced order model (ROM) of the function sinc(x)=sin(x)/x

# *** The following class demonstrates the elements needed to define a model for Rombus ***
#
# It must do the following:
#    1. inherit from the RombusModel Class (which is imported above);
#    2. call `coordinate.set()` to define the domain on which the ROM will be defined;
#    3. call `ordinate.set()` to define the name and type of the space the model will
#       map the domain onto
#    4. include one-or-more calls to 'params.add_param()' to define the names and
#       limits of the parameters of the model; over-and-on
#    5. define a 'compute' method which takes a parameter set as a named tuple, a
#       domain specified as a numpy array, and returns a numpy array of the type
#       specified by `ordinate.set()`.


class Model(RombusModel):
    """Class for creating a ROM for the function sinc(x)=sin(x)/x"""

    # Set the domain over-and-on which the ROM will be defined
    # coordinate.set() takes a name, a minimum value, a maximum value,
    # an integer number of values and optional keyword values 'label' -
    # which is used for plots, etc. - and dtype, which must match the type
    # of min_value and max_value
    coordinate.set("x", 0.0, 100.0, 1024, label="$x$", dtype=np.dtype("float64"))  # type: ignore # noqa F821

    # Set the ordinate the model will map the domain to
    # ordinate.set() takes a name and optional keyword values 'label' -
    # which is used for plots, etc. - and dtype, which must match the type
    # returned by the compute() method defined below.
    ordinate.set("y", label="$sinc(x)$", dtype=np.dtype("float64"))  # type: ignore # noqa F821

    # Add as many calls to params.add() as your model has parameters.
    # Samples will be sent to the 'compute()' method (see below) as
    # named tuples and will be addressable using the names you give them.
    #
    # Note that you may need to silence linting errors by adding a '# noqa F821'
    # directive at the end.  params is initialised, it's just done in a way that
    # IDEs and linters have troubles following.
    #
    # Syntax: params.add(name, min_value_allowed,max_value_allowed)
    params.add("A", 0, 10)  # type: ignore # noqa F821

    # Compute the model.  This function should accept a named tuple with
    # member names given by the params.add() calls above and should
    # return a numpy array of type given my ordinate.set() above
    def compute(self, params: NamedTuple, x) -> np.ndarray:
        """Compute the model for a given parameter set."""

        return sinc_vectorized(params.A * x)  # type: ignore


# Create the function that will compute our model
def sinc_scalar(x):
    """A scalar version of our model function, which takes a
    value x and returns a value x."""

    if x == 0.0:
        return 1.0
    else:
        return math.sin(x) / x


# Use np.vectorize() to create a function that accepts a
# numpy array and returns a numpy array.  This will be the
# function the model calls to do the actual work
sinc_vectorized = np.vectorize(sinc_scalar)
