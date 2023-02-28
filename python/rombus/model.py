import importlib
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import Any, List
from tqdm.auto import tqdm

import numpy as np

import rombus.mpi as mpi


def init_model(model: str):
    # Import the model code
    model_class = import_from_string(model)
    return model_class()


class RombusModel(metaclass=ABCMeta):
    def __init__(self):

        # Run an optional init() method
        self.init()

        # Ensure params is a list of strings
        assert bool(self.params) and all(isinstance(elem, str) for elem in self.params)

        # Ensure that model_dtype is a string
        assert type(self.model_dtype) == str

        # Create the named tuple that will be used for parameters
        self.params_dtype = namedtuple("params_dtype", self.params)

    def init(self):
        pass

    @property
    @abstractmethod  # make sure this is the inner-most decorator
    def model_dtype(self):
        pass

    @property
    @abstractmethod  # make sure this is the inner-most decorator
    def params(self):
        pass

    @abstractmethod  # make sure this is the inner-most decorator
    def init_domain(self):
        pass

    @abstractmethod  # make sure this is the inner-most decorator
    def compute(self, params: np.ndarray, domain) -> np.ndarray:
        pass


    def generate_model_set(self, points: List[np.ndarray]) -> np.ndarray:
        """returns a list of models (one for each row in 'points')"""

        domain = self.init_domain()

        my_ts = np.zeros(shape=(len(points), len(domain)), dtype=self.model_dtype)
        for i, params_numpy in enumerate(
            tqdm(points, desc=f"Generating training set for rank {mpi.RANK}")):
            params = self.params_dtype(
                **dict(zip(self.params, np.atleast_1d(params_numpy)))
            )
            model_i = self.compute(params, domain)
            if self.model_dtype == complex:
                my_ts[i] = model_i / np.sqrt(np.vdot(model_i, model_i))
            else:
                my_ts[i] = model_i / np.sqrt(np.dot(model_i, model_i))

        return my_ts


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
