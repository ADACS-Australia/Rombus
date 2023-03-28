Rombus: Helps you qucikly and easily compute slow and complex models
====================================================================

Rombus is a tool for building reduced order models (ROMs): matrix representations of arbitrary
models which can be rapidly and easily computed for arbitrary parameter sets.

Building a ROM with Rombus is easy.  All you need to do is define your model like this (in this trivial case, a file named `my_model.py` specifying a simple second-order polynomial):
```
from numpy import ndarray, polyval, linspace
from rombus.model import RombusModel
from typing import NamedTuple


class model(RombusModel):
    """Class for creating a ROM for the function y=a2*x^2+a1*x+a0"""

    params.add("a0", -10, 10)
    params.add("a1", -10, 10)
    params.add("a2", -10, 10)

    # Set the domain over-and-on which the ROM will be defined
    def set_domain(self) -> ndarray:
        return linspace(0, 10, 11)

    def compute(self, p: NamedTuple, x: ndarray) -> ndarray:
        """Compute the model for a given parameter set."""
        return polyval([p.a2, p.a1, p.a0], x)
```
and specify a set of points (in this case, the file `my_model_samples.py`) to build your ROM from:
```
-10, -10,-10
-10,  10,-10
-10, -10, 10
-10,  10, 10
 10, -10,-10
 10,  10,-10
 10, -10, 10
 10,  10, 10
```
You build your ROM like this:
```
$ rombus build my_model:model my_model_samples.csv
```
This produces an _HDF5_ file named `my_model.hdf5`.  You can then use your new ROM in
your Python projects like this:
```
from rombus.rom import ReducedOrderModel

ROM = ReducedOrderModel.from_file('my_model.hdf5')
sample = ROM.model.sample({"a0":0,"a1":0,"a2":1})
model_ROM = ROM.evaluate(sample)
for x, y in zip(ROM.model.domain,model_ROM):
    print(f"{x:.2f} {y:.2f}")
```
which generates the output:
```
0.00 0.00
1.00 1.00
2.00 4.00
3.00 9.00
4.00 16.00
5.00 25.00
6.00 36.00
7.00 49.00
8.00 64.00
9.00 81.00
10.00 100.00
```
