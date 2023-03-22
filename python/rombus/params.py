from typing import Callable, List, NamedTuple, Optional

import numpy as np

MAX_N_TRIES = 1000


class Params(object):
    def __init__(self) -> None:
        self.params: List[NamedTuple] = []
        self.names: List[str] = []
        self.count: int = 0
        self._idx: int = -1
        self.params_dtype: Optional[NamedTuple] = None
        self._validate: Callable = lambda x: True

    def add(
        self, name: str, range_min: float, range_max: float, dtype: type = float
    ) -> None:

        # N.B.: mypy struggles with NamedTuples, so typing is turned off for the following
        Param = NamedTuple(f"Param{len(self.params)}", [("name", str), ("min", dtype), ("max", dtype), ("dtype", dtype)])  # type: ignore
        self.params.append(Param(name, dtype(range_min), dtype(range_max), dtype))  # type: ignore
        self.count = self.count + 1
        self.names.append(name)

        # Update the named tuple that will be used to convert numpy arrays to Param objects
        # N.B.: mypy struggles with NamedTuples, so typing is turned off for the following
        self.params_dtype = NamedTuple("params_dtype", [(name, dtype) for name in self.names])  # type: ignore

    def set_validation(self, func: Callable) -> None:
        self._validate = func

    def generate_random_sample(
        self, random_generator: np.random._generator.Generator
    ) -> np.ndarray:

        new_sample: np.ndarray = np.ndarray(self.count, dtype=np.float64)
        n_tries = 0
        while True:
            for i, param in enumerate(self.params):
                # N.B.: mypy struggles with NamedTuples, so typing is turned off for this next line
                new_sample[i] = random_generator.uniform(low=param.min, high=param.max)  # type: ignore
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

    def np2param(self, params_np: np.ndarray) -> NamedTuple:
        if not self.params_dtype:
            return NamedTuple("params_empty", [])
        else:
            # N.B.: mypy struggles with NamedTuples, so typing is turned off for this next line
            return self.params_dtype(**dict(zip(self.names, np.atleast_1d(params_np))))  # type: ignore

    def __iter__(self):
        self._idx = -1
        return self

    def __next__(self):
        self._idx = self._idx + 1
        if self._idx >= self.count:
            raise StopIteration
        return self.params[self._idx]

    def __len__(self):
        return self.count
