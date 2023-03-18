from collections import namedtuple
from typing import Callable

import numpy as np

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
