from numpy import ndarray, polyval
from rombus.model import RombusModel
from typing import NamedTuple


class Model(RombusModel):
    """Class for creating a ROM for the function y(x)=a2*x^2+a1*x+a0"""

    # the x-range for the model
    coordinate.set("x", 0.0, 10.0, 11, label="$x$")

    # the label for what you are trying to model
    ordinate.set("y", label="$y(x)$")

    # the parameters passed to the model and their ranges
    params.add("a0", -10, 10)
    params.add("a1", -10, 10)
    params.add("a2", -10, 10)

    def compute(self, p: NamedTuple, x: ndarray) -> ndarray:
        """Compute the model for a given parameter set."""
        return polyval([p.a2, p.a1, p.a0], x)
