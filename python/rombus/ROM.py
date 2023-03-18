import timeit

import h5py
import mpi4py
import numpy as np

import rombus._core.mpi as mpi
from rombus.model import RombusModel
from rombus.samples import Samples
from rombus.empirical_interpolant import EmpiricalInterpolant
from rombus.reduced_basis import ReducedBasis

DEFAULT_TOLERANCE = 1e-14
DEFAULT_REFINE_N_RANDOM = 100


class ReducedOrderModel(object):
    def __init__(
        self,
        model,
        samples,
        reduced_basis=None,
        empirical_interpolant=None,
        tol=DEFAULT_TOLERANCE,
    ):

        if isinstance(model, str):
            self.model = RombusModel.load(model)
        elif isinstance(model, RombusModel):
            self.model = model
        else:
            raise Exception

        self.model = model
        self.samples = samples
        self.reduced_basis = reduced_basis
        self.empirical_interpolant = empirical_interpolant

    def build(self, do_step=None, tol=DEFAULT_TOLERANCE):

        if do_step is None or do_step == "RB":
            self.reduced_basis = ReducedBasis().compute(
                self.model, self.samples, tol=tol
            )

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
    def from_file(cls, file_in):
        """Create a ROM instance from a file"""
        close_file = False
        if not isinstance(file_in, str):
            h5file = file_in
        else:
            h5file = h5py.File(file_in, "r")
            close_file = True
        model = RombusModel.from_file(h5file)
        samples = Samples.from_file(h5file)
        reduced_basis = ReducedBasis.from_file(h5file)
        empirical_interpolant = EmpiricalInterpolant.from_file(h5file)
        if close_file:
            h5file.close()
        return cls(
            model,
            samples,
            reduced_basis=reduced_basis,
            empirical_interpolant=empirical_interpolant,
        )

    def timing(self, samples):
        start_time = timeit.default_timer()
        for i, sample in enumerate(samples.samples):
            params_numpy = self.model.params.np2param(sample)
            _ = self.evaluate(params_numpy)
        return timeit.default_timer() - start_time

    def _validate_and_refine_basis(self, n_random, tol=DEFAULT_TOLERANCE, iterate=True):

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
