import importlib
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import Any

import numpy as np


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
