from typing import List

import h5py
import numpy as np

import rombus._core.mpi as mpi

from rombus.model import RombusModel

DEFAULT_TOLERANCE = 1e-14
DEFAULT_REFINE_N_RANDOM = 100


class Samples(object):
    def __init__(self, model, filename=None, n_random=0):

        self.model = model

        # Needed for random number generation
        self.n_random = n_random
        self.random = None
        self.random_starting_state = None

        # Initialise samples
        self.n_samples = 0
        self.samples = []
        if filename:
            self._add_from_file(filename)
        if self.n_random > 0:
            self._add_random_samples(self.n_random)

    def extend(self, new_samples):

        self.samples.extend(new_samples)
        self.n_samples = self.n_samples + len(new_samples)

    def write(self, h5file):
        """Save samples to file"""

        h5_group = h5file.create_group("samples")
        self.model.write(h5_group)
        h5_group.create_dataset("samples", data=self.samples)
        h5_group.create_dataset("n_samples", data=self.n_samples)

    @classmethod
    def from_file(cls, file_in):
        """Create a ROM instance from a file"""

        close_file = False
        if not isinstance(file_in, str):
            h5file = file_in
        else:
            h5file = h5py.File(file_in, "r")
            close_file = True

        model_str = h5file["samples/model/model_str"].asstr()[()]
        model = RombusModel.load(model_str)
        samples = cls(model)
        samples.samples = [np.array(x) for x in h5file["samples/samples"]]
        samples.n_samples = np.int32(h5file["samples/n_samples"])
        if close_file:
            h5file.close()
        return samples

    def _add_from_file(self, filename_in: str):

        # dividing greedypoints into chunks
        if mpi.RANK_IS_MAIN:
            if filename_in.endswith(".npy"):
                samples = np.load(filename_in)
            elif filename_in.endswith(".csv"):
                samples = [
                    np.atleast_1d(x)
                    for x in np.genfromtxt(filename_in, delimiter=",", comments="#")
                ]
            else:
                raise Exception
        else:
            samples = None

        new_samples = self._decompose_samples(samples)
        n_new_samples = len(new_samples)

        self.samples.extend(new_samples)
        self.n_samples = self.n_samples + n_new_samples

    def _add_random_samples(self, n_samples):

        self.random = np.random.default_rng()
        self.random_starting_state = np.random.get_state()
        samples = []
        for _ in range(n_samples):
            # new_sample = np.ndarray(self.model.params.count, dtype=np.float64)
            # for i, param in enumerate(self.model.params):
            #    new_sample[i] = self.random.uniform(low=param.min, high=param.max)
            new_sample = self.model.params.generate_random_sample(self.random)
            samples.append(new_sample)

        new_samples = self._decompose_samples(samples)
        n_new_samples = len(new_samples)

        self.samples.extend(new_samples)
        self.n_samples = self.n_samples + n_new_samples

    def _decompose_samples(
        self,
        samples: List[np.ndarray],
    ):
        chunks = None
        if mpi.RANK_IS_MAIN:
            chunks = [[] for _ in range(mpi.SIZE)]
            for i, chunk in enumerate(samples):
                chunks[i % mpi.SIZE].append(chunk)

        return mpi.COMM.scatter(chunks, root=mpi.MAIN_RANK)
