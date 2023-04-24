Getting Started
===============

Getting started with Rombus is easy.  To install it you just need to run the following command:
```console
$ pip install rombus
```
Once that's done, you need just two things to build a Reduced Order Model (ROM):
1. an ___appropriately defined Python class___ (more on that below) placed somewhere in your Python path or in your current working directory, and
2. a ___file listing a set of parameter samples___ to be used to construct the model.

Let's give this a try for the trivial example of the function: $sinc(Ax)=sin(Ax)/(Ax)$
:::{tip}
The files used in what follows can be easily generated using the following quickstart command:
```sh
$ rombus quickstart my_project
```
This is a good way to quickly generate a project template which you can then modify for your new Rombus projects.
:::
First we need to define a model class.  We place the following code into a file called `my_project.py`:
```Python
import numpy as np
from rombus.model import RombusModel
from typing import NamedTuple
import math

# Your model class must do the following:
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
    coordinate.set("x", 0.0, 100.0, 1024, label="$x$", dtype=np.dtype("float64")  # type: ignore # noqa F821

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
```
Next, we need to define a set of parameter samples for which our model will be computed, the results of which will be used to
compute our ROM.  Let's place the following in a `my_project.csv` file:
```
0.1
0.5
1.0
2.0
5.0
10.0
```
Now we have everything we need to build a first iteration of our ROM.  This can be done with the Rombus CLI application as
follows:
```sh
$ rombus build sinc:model sinc_samples.csv
```
::: {note}
Note the syntax for specifying the model: **sub.module.name:ClassName**.  You should omit the `.py` from the filename.
:::
This should generate your ROM and place it in an _.hdf5_ file named `my_project.hdf5`.  Let's see how accurate it is:
```console
$ rombus evaluate sinc.hdf5 2.0
```
This should place a plot in a file named `comparison.pdf` which looks similar to the following:
![Figure 1](assets/comparison_sinc_2pt0.pdf){align=center}

The agreement looks impressive, until you notice that $A=2.0$ is one of the parameter samples given above and know that ROMs are
constructed such that they are machine-level accurate for any set of parameters used to build them.  Let's try a value that we did
*not* use to build the model:
```console
$ rombus evaluate sinc.hdf5 3.5
```
This should generate the following plot:
![Figure 2](assets/comparison_sinc_3pt5.pdf){align=center}

Yikes.  This is not a good agreement!

Let's see if we can do better by refining the model:
```console
$ rombus refine my_project.hdf5
```
This instructs Rombus to iterate through sets of randomly generated parameter samples and write a new ROM to the file
`my_project_refined.hdf5`.  Let's take a look at this new model:
```console
$ rombus evaluate my_project_refined.hdf5 3.5
```
This should generate the following plot:
![Figure 3](assets/comparison_sinc_refined_3pt5.pdf){align=center}
Much better!

One last thing: let's see how fast our ROM is relative to our original model.  We can do this by running the following:
```console
$ rombus timing my_project.hdf5

Timing information for ROM:   1.22e-03s for 100 calls (1.22e-05 per sample).
Timing information for model: 1.13e-02s for 100 calls (1.13e-04 per sample).
ROM is 9.28X faster than the source model.
```

```console
$ rombus timing my_project_refined.hdf5

Timing information for ROM:   1.41e-01s for 100 calls (1.41e-03 per sample).
Timing information for model: 1.38e-02s for 100 calls (1.38e-04 per sample).
ROM is 10.16X slower than the source model.
```
**Now, go ahead and try to generate your own model.  Just modify `my_project.py` and `my_project_samples.csv` to suit your needs and repeat
the steps above.**
::: {note}
Usually, Rombus won't be used to generate representations of trivial models such as this.  It excels at furnishing representations of models
which can take minites-to-weeks of CPU time to generate.  Once a ROM is successfuly built for such a model, analyses can be performed with
many orders of magnitude of speed-up.
:::
