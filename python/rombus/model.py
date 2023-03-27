import importlib
import sys
import os
import shutil
import timeit
from abc import ABCMeta, abstractmethod
from collections import Counter
from typing import Any, Dict, Self, Tuple, TYPE_CHECKING
from tqdm.auto import tqdm  # type: ignore

import numpy as np

import rombus._core.mpi as mpi
import rombus._core.hdf5 as hdf5
import rombus.exceptions as exceptions
from rombus.params import Params
from typing import NamedTuple

# Need to put Samples in quotes below and check TYPE_CHECKING here to
# manage circular imports with models.py
if TYPE_CHECKING:
    from rombus.samples import Samples


class _RombusModelMeta(type):
    def __prepare__(name, *args, **kwargs):
        """Initialise the dictionary that gets passed to __new___.

        This is needed here because we don't want the user to have to
        initialise the member(s) that we are adding.  This is the only
        method that gets sourced before the class code is executed, so
        it needs to be done here, not in __new___.
        """

        result = dict()

        # Initialise the following members↵
        result["params"] = Params()
        result["model_dtype"] = np.dtype("float64")

        return result

    def __new__(mcs, name, bases, dct):

        # Perform super-metaclass construction↵
        return super(_RombusModelMeta, mcs).__new__(mcs, name, bases, dct)


class _RombusModelABCMeta(_RombusModelMeta, ABCMeta):
    pass


class RombusModel(metaclass=_RombusModelABCMeta):
    """Baseclass from which all RombusModels must inherit.

    Attributes
    ----------
    model_str : String from which the model was originally instantiated
    model_basename : The submodule where the model was found.  Used as file basename.
    domain : The domain used for the model
    n_domain : Number of elements in the domain
    """

    # These two members are instantiated by the metaclass
    params: Params
    model_dtype: type | np.dtype
    domain: np.ndarray
    n_domain: int

    def __init__(self, model: str):

        # Keep track of the model string so we can reinstantiate from a saved state
        self.model_str: str = model
        self.model_basename: str = self.model_str.split(":")[0].split(".")[-1]

        # Run an optional cache() method
        self.cache()

        # Initialise the domain
        self.domain: np.ndarray = self.set_domain()
        self.n_domain: int = len(self.domain)

        # Check that the domain has been suitably set
        assert self.n_domain > 0

        # Check that at least one parameter has beed defined
        if self.params.count <= 0:
            raise exceptions.RombusModelParamsError(
                f"Invalid number of parameters ({self.params.count}) specified for Rombus model ({self})."
            )

    def __str__(self):
        return f"<RombusModel from {self.model_str}>"

    @abstractmethod  # make sure this is the inner-most decorator
    def set_domain(self) -> np.ndarray:
        pass

    @abstractmethod  # make sure this is the inner-most decorator
    def compute(self, params: NamedTuple, domain: np.ndarray) -> np.ndarray:
        pass

    def cache(self):
        pass

    # Need to put Samples in quotes and check TYPE_CHECKING above to manage circular import with models.py
    def generate_model_set(self, samples: "Samples") -> np.ndarray:
        """returns a list of models (one for each row in 'samples')"""

        my_ts: np.ndarray = np.zeros(
            shape=(samples.n_samples, self.n_domain), dtype=self.model_dtype
        )
        for i, params_numpy in enumerate(
            tqdm(samples.samples, desc=f"Generating training set for rank {mpi.RANK}")
        ):
            model_i = self.compute(self.params.np2param(params_numpy), self.domain)
            my_ts[i] = model_i / np.sqrt(np.vdot(model_i, model_i))

        return my_ts

    def parse_cli_params(self, args: Tuple[str, ...]) -> Dict[str, Any]:
        """
        Parse parameters of the from param0=val0, param1=val1, ... to a
        dictionary.

        Generally used to parse the optional arguments recieved from Click
        into a format that can be converted into a Params or Numpy object

        Parameters
        ----------
        args
            [TODO:description]

        Returns
        -------
        Dict[str, Any]
            [TODO:description]

        Raises
        ------
        Exception:
            [TODO:description]
        Exception:
            [TODO:description]
        """
        model_params = dict()
        for param_i in args:
            if not param_i.startswith("-"):
                res = param_i.split("=")
                if len(res) == 2:
                    # NOTE: for now, all parameters are assumed to be floats
                    model_params[res[0]] = float(res[1])
                else:
                    raise Exception(f"Don't know what to do with argument '{param_i}'")
            else:
                raise Exception(f"Don't know what to do with option '{param_i}'")

        # Check that all parameters are specified and that they match what is
        # defined in the model
        assert Counter(model_params.keys()) == Counter(self.params.names)

        return model_params

    def sample(self, kwargs: Dict[str, Any]) -> NamedTuple:
        return self.params.params_dtype(**kwargs)  # type: ignore

    def timing(self, samples: "Samples") -> float:
        start_time = timeit.default_timer()
        for i, sample in enumerate(samples.samples):
            params_numpy = self.params.np2param(sample)
            _ = self.compute(params_numpy, self.domain)
        return timeit.default_timer() - start_time

    @classmethod
    def load(cls, model: str | Self) -> Self:

        if isinstance(model, str):
            try:
                model_class = _import_from_string(model)
            except exceptions.RombusException as e:
                exceptions.handle_exception(e)
            else:
                return model_class(model)
        elif not isinstance(model, RombusModel):
            raise exceptions.RombusModelInitError(
                "Invalid type ({type(model)}) specified when loading model {model}."
            )
        return model  # type: ignore

    def write(self, h5file: hdf5.File) -> None:
        """Save samples to file"""

        try:
            h5_group = h5file.create_group("model")
            h5_group.create_dataset("model_str", data=self.model_str)
        except IOError as e:
            exceptions.handle_exception(e)

    @classmethod
    def from_file(cls, file_in: hdf5.FileOrFilename) -> Self:
        """Create a ROM instance from a file"""

        try:
            h5file, close_file = hdf5.ensure_open(file_in)
            model_str = h5file["model/model_str"].asstr()[()]
            if close_file:
                h5file.close()
        except IOError as e:
            exceptions.handle_exception(e)
        return cls.load(model_str)

    @classmethod
    def write_project_template(cls, project_name: str) -> None:
        """Write a project model and sample file to start a new project from."""

        # Set the model we will template from
        model_name = "sinc"

        # Set source file paths
        pkgdir = sys.modules["rombus"].__path__[0]
        model_file_source = os.path.join(pkgdir, "models", f"{model_name}.py")
        samples_file_source = os.path.join(
            pkgdir, "models", f"{model_name}_samples.csv"
        )

        # Set output file paths
        model_file_out = os.path.join(os.getcwd(), f"{project_name}.py")
        samples_file_out = os.path.join(os.getcwd(), f"{project_name}_samples.csv")

        # Copy files
        try:
            shutil.copy(model_file_source, model_file_out)
            shutil.copy(samples_file_source, samples_file_out)
        except IOError as e:
            exceptions.handle_exception(e)


RombusModelType = RombusModel | str

# The code that follows is modified from code copied from the Uvicorn codebase:
#     https://github.com/encode/uvicorn (commit: d613cbea388bafafb6f642077c035ed137deea61)
# Copyright © 2017-present, [Encode OSS Ltd](https://www.encode.io/).
# All rights reserved.
def _import_from_string(import_str: str) -> Any:

    if not isinstance(import_str, str):
        raise exceptions.RombusModelImportFromStringError(
            f'Import string must be a string with format "<module>:<attribute>".  It is actually of type {type(import_str)}.'
        )

    # Make sure the CWD is in the import path
    sys.path.append(os.getcwd())
    sys.path = list(dict.fromkeys(sys.path))

    # Split the string
    module_str, _, attrs_str = import_str.partition(":")
    if not module_str or not attrs_str:
        raise exceptions.RombusModelImportFromStringError(
            f'Import string "{import_str}" must be in format "<module>:<attribute>".'
        )

    # Try to import the module
    try:
        module = importlib.import_module(module_str)
    except ImportError as exc:
        if exc.name != module_str:
            raise exc from None
        raise exceptions.RombusModelImportFromStringError(
            f'Could not import module "{module_str}".\n'
        )
    instance = module

    # Try to grab the specified class
    try:
        for attr_str in attrs_str.split("."):
            instance = getattr(instance, attr_str)
    except AttributeError:
        raise exceptions.RombusModelImportFromStringError(
            f'Attribute "{attrs_str}" not found in module "{module_str}".'
        )

    return instance
