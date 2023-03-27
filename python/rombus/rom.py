import timeit


import h5py  # type: ignore
import mpi4py
import numpy as np

from typing import Optional, Self

import rombus._core.mpi as mpi
import rombus._core.hdf5 as hdf5
import rombus.exceptions as exceptions
from rombus.model import RombusModel, RombusModelType
from rombus.samples import Samples
from rombus.ei import EmpiricalInterpolant
from rombus.reduced_basis import ReducedBasis

DEFAULT_TOLERANCE: float = 1e-14
DEFAULT_REFINE_N_RANDOM: int = 100


class ReducedOrderModel(object):
    def __init__(
        self,
        model: RombusModelType,
        samples: Samples,
        reduced_basis: Optional[ReducedBasis] = None,
        empirical_interpolant: Optional[EmpiricalInterpolant] = None,
        tol: float = DEFAULT_TOLERANCE,
    ):

        self.model: RombusModel = RombusModel.load(model)

        self.samples = samples
        self.reduced_basis = reduced_basis
        self.empirical_interpolant = empirical_interpolant

    def build(self, do_step=None, tol=DEFAULT_TOLERANCE):

        if do_step is None or do_step == "RB":
            try:
                self.reduced_basis = ReducedBasis().compute(
                    self.model, self.samples, tol=tol
                )
            except exceptions.RombusException as e:
                e.handle_exception()

        if do_step is None or do_step == "EI":
            if self.reduced_basis is None:
                raise Exception
            self.empirical_interpolant = EmpiricalInterpolant().compute(
                self.reduced_basis
            )

        return self

    def refine(
        self, n_random=DEFAULT_REFINE_N_RANDOM, tol=DEFAULT_TOLERANCE, iterate=True
    ):

        if self.reduced_basis is None:
            self.reduced_basis = ReducedBasis().compute(
                self.model, Samples(self.model, n_random=n_random), tol=tol
            )
        self._validate_and_refine_basis(n_random, tol=tol, iterate=iterate)

        self.empirical_interpolant = EmpiricalInterpolant().compute(self.reduced_basis)

        return self

    def evaluate(self, params):
        _signal_at_nodes = self.model.compute(params, self.empirical_interpolant.nodes)
        return np.dot(_signal_at_nodes, np.real(self.empirical_interpolant.B_matrix))

    def write(self, filename):
        """Save ROM to file"""
        with h5py.File(filename, "w") as h5file:
            self.model.write(h5file)
            self.samples.write(h5file)
            if self.reduced_basis is not None:
                self.reduced_basis.write(h5file)
            if self.empirical_interpolant is not None:
                self.empirical_interpolant.write(h5file)

    @classmethod
    def from_file(cls, file_in: hdf5.FileOrFilename) -> Self:
        """Create a ROM instance from a file.

        If given an open HDF5 file, then use it.  If given a valid filename then open it for reading but close it when done."""

        h5file, close_file = hdf5.ensure_open(file_in)

        model: RombusModel = RombusModel.from_file(h5file)
        samples: Samples = Samples.from_file(h5file)
        reduced_basis: ReducedBasis = ReducedBasis.from_file(h5file)
        empirical_interpolant: EmpiricalInterpolant = EmpiricalInterpolant.from_file(
            h5file
        )

        if close_file:
            h5file.close()

        return cls(
            model,
            samples,
            reduced_basis=reduced_basis,
            empirical_interpolant=empirical_interpolant,
        )

    def timing(self, samples: Samples) -> float:
        start_time = timeit.default_timer()
        for i, sample in enumerate(samples.samples):
            params_numpy = self.model.params.np2param(sample)
            _ = self.evaluate(params_numpy)
        return timeit.default_timer() - start_time

    def _validate_and_refine_basis(
        self, n_random: int, tol: float = DEFAULT_TOLERANCE, iterate: bool = True
    ) -> None:

        if not self.reduced_basis:
            self.reduced_basis = ReducedBasis().compute(
                self.model, self.samples, tol=tol
            )

        n_selected_greedy_points_global = np.iinfo(np.int32).max
        n_greedy_last = len(self.reduced_basis.greedypoints)
        n_greedy_last_global = mpi.COMM.allreduce(n_greedy_last, op=mpi4py.MPI.SUM)
        while True:
            # generate validation set by randomly sampling the parameter space
            new_samples = Samples(self.model, n_random=n_random)
            my_vs = self.model.generate_model_set(new_samples)

            # test validation set
            RB_transpose = np.transpose(self.reduced_basis.matrix)
            selected_greedy_points = []
            for i, validation_sample in enumerate(new_samples.samples):
                if self.model.model_dtype == complex:
                    proj_error = 1 - np.sum(
                        [
                            np.real(np.conjugate(d_i) * d_i)
                            for d_i in np.dot(my_vs[i], RB_transpose)
                        ]
                    )
                else:
                    proj_error = 1 - np.sum(np.dot(my_vs[i], RB_transpose) ** 2)
                if proj_error > tol:
                    selected_greedy_points.append(validation_sample)
            n_selected_greedy_points_global = mpi.COMM.allreduce(
                len(selected_greedy_points), op=mpi4py.MPI.SUM
            )
            if mpi.RANK_IS_MAIN:
                print(f"Number of samples added: {n_selected_greedy_points_global}")

            # add the inaccurate points to the original selected greedy
            # points and remake the basis
            self.samples.extend(selected_greedy_points)
            self.reduced_basis = ReducedBasis().compute(
                self.model, self.samples, tol=tol
            )
            n_greedy_new = len(self.reduced_basis.greedypoints)
            n_greedy_new_global = mpi.COMM.allreduce(n_greedy_new, op=mpi4py.MPI.SUM)

            if not iterate or n_greedy_new_global == n_greedy_last_global:
                break
            else:
                if mpi.RANK_IS_MAIN:
                    print(
                        f"Current number of accepted greedy points: {n_greedy_new_global}"
                    )
                n_greedy_last_global = n_greedy_new_global
