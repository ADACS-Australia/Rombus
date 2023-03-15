import importlib
import sys
import os
import shutil
from abc import ABCMeta, abstractmethod
from collections import namedtuple, Counter
from typing import Any, List, Callable
from tqdm.auto import tqdm

import h5py
import numpy as np

import rombus._core.mpi as mpi

MAX_N_TRIES = 1000


class Params(object):
    def __init__(self):
        self.params = []
        self.names = []
        self.count = 0
        self._idx = -1
        self.params_dtype = None
        self._validate = lambda x: True

    def add(self, name, range_min, range_max):
        Param = namedtuple("Param", "name min max")
        self.params.append(Param(name, range_min, range_max))
        self.count = self.count + 1
        self.names.append(name)

        # Update the named tuple that will be used to convert numpy arrays to Param objects
        self.params_dtype = namedtuple("params_dtype", self.names)

    def set_validation(self, func: Callable):
        self._validate = func

    def generate_random_sample(self, random_generator):

        new_sample = np.ndarray(self.count, dtype=np.float64)
        n_tries = 0
        while True:
            for i, param in enumerate(self.params):
                new_sample[i] = random_generator.uniform(low=param.min, high=param.max)
            param = self.np2param(new_sample)
            if self._validate(param):
                break
            else:
                n_tries = n_tries + 1
                if n_tries >= MAX_N_TRIES:
                    raise Exception(
                        f"Max number of tries ({MAX_N_TRIES}) reached when trying to generate a valid random parameter set"
                    )
        return new_sample

    def np2param(self, params_np):
        return self.params_dtype(**dict(zip(self.names, np.atleast_1d(params_np))))

    def __iter__(self):
        self._idx = -1
        return self

    def __next__(self):
        self._idx = self._idx + 1
        if self._idx >= self.count:
            raise StopIteration
        return self.params[self._idx]


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
    def __init__(self, model_str):

        # Check that at least one parameter has beed defined
        assert self.params.count > 0

        # Ensure that model_dtype is of the right type
        # assert self.model_dtype.type == np.dtype

        # Run an optional init() method
        self.cache()

        # Initialise the domain
        self.domain = self.set_domain()
        self.n_domain = len(self.domain)

        # Check that the domain has been suitably set
        assert self.n_domain > 0

        # Keep track of the model string so we can reinstantiate from a saved state
        self.model_str = model_str
        self.model_basename = self.model_str.split(":")[0].split(".")[-1]

    def cache(self):
        pass

    @abstractmethod  # make sure this is the inner-most decorator
    def set_domain(self):
        pass

    @abstractmethod  # make sure this is the inner-most decorator
    def compute(self, params: np.ndarray, domain) -> np.ndarray:
        pass

    def generate_model_set(self, samples: List[np.ndarray]) -> np.ndarray:
        """returns a list of models (one for each row in 'samples')"""

        my_ts = np.zeros(
            shape=(samples.n_samples, self.n_domain), dtype=self.model_dtype
        )
        for i, params_numpy in enumerate(
            tqdm(samples.samples, desc=f"Generating training set for rank {mpi.RANK}")
        ):
            model_i = self.compute(self.params.np2param(params_numpy), self.domain)
            if self.model_dtype == complex:
                my_ts[i] = model_i / np.sqrt(np.vdot(model_i, model_i))
            else:
                my_ts[i] = model_i / np.sqrt(np.dot(model_i, model_i))

        return my_ts

    def parse_cli_params(self, args):
        """Parse parameters from arguments given on command line"""

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

    @classmethod
    def load(cls, model: str):
        """Import the model code"""

        model_class = import_from_string(model)
        return model_class(model)

    def write(self, h5file):
        """Save samples to file"""

        h5_group = h5file.create_group("model")
        h5_group.create_dataset("model_str", data=self.model_str)

    @classmethod
    def from_file(cls, file_in):
        """Create a ROM instance from a file"""

        close_file = False
        if not isinstance(file_in, str):
            h5file = file_in
        else:
            h5file = h5py.File(file_in, "r")
            close_file = True

        model_str = h5file["model/model_str"].asstr()[()]
        if close_file:
            h5file.close()
        return cls.load(model_str)

    @classmethod
    def write_project_template(cls, project_name):
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
        shutil.copy(model_file_source, model_file_out)
        shutil.copy(samples_file_source, samples_file_out)


# The code that follows has been copied directly from the Uvicorn codebase: https://github.com/encode/uvicorn
# (commit: d613cbea388bafafb6f642077c035ed137deea61)
#
# Copyright © 2017-present, [Encode OSS Ltd](https://www.encode.io/).
# All rights reserved.


class ImportFromStringError(Exception):
    pass


def import_from_string(import_str: Any) -> Any:
    if not isinstance(import_str, str):
        return import_str

    module_str, _, attrs_str = import_str.partition(":")
    if not module_str or not attrs_str:
        message = (
            'Import string "{import_str}" must be in format "<module>:<attribute>".'
        )
        raise ImportFromStringError(message.format(import_str=import_str))

    try:
        module = importlib.import_module(module_str)
    except ImportError as exc:
        if exc.name != module_str:
            raise exc from None
        message = 'Could not import module "{module_str}".'
        raise ImportFromStringError(message.format(module_str=module_str))

    instance = module
    try:
        for attr_str in attrs_str.split("."):
            instance = getattr(instance, attr_str)
    except AttributeError:
        message = 'Attribute "{attrs_str}" not found in module "{module_str}".'
        raise ImportFromStringError(
            message.format(attrs_str=attrs_str, module_str=module_str)
        )

    return instance
