import sys
from typing import List

import h5py
import mpi4py
import numpy as np

import rombus._core.algorithms as algorithms
import rombus._core.misc as misc
import rombus._core.mpi as mpi
from rombus.model import RombusModel

DEFAULT_TOLERANCE = 1e-14
DEFAULT_REFINE_N_RANDOM = 100


class ROM(object):
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
                self.model, Samples(self.model, n_random=n_random, tol=tol)
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


class ReducedBasis(object):
    def __init__(self, matrix=[], greedypoints=[], error_list=[]):
        """Make reduced basis

        FILENAME_IN is the 'greedy points' numpy file to take as input
        """

        self.matrix = matrix
        self.greedypoints = greedypoints
        self.error_list = error_list

    def compute(self, model, samples, tol=DEFAULT_TOLERANCE):

        if isinstance(model, str):
            self.model = RombusModel.load(model)
        elif isinstance(model, RombusModel):
            self.model = model
        else:
            raise Exception

        # Compute the model for each given sample
        my_ts = model.generate_model_set(samples)

        if mpi.RANK_IS_MAIN:
            print("Filling basis with greedy-algorithm")

        # hardcoding 1st model to be used to start the basis
        self._init_matrix(my_ts[0])
        basis_indicies = [0]

        # NOTE: NEED TO FIX; THE FIRST INDEX IS BEING ADDED TWICE HERE!
        error = np.inf
        iter = 1
        pc_matrix = []
        while error > tol:
            self.matrix, pc_matrix, error_data = self._add_next_model_to_basis(
                pc_matrix, my_ts, iter
            )
            err_rnk, err_idx, error = error_data

            # log and cache some data
            m = f">>> Iter {iter:003}: err {error:.1E} (rank {err_rnk:002}@idx{err_idx:003})"
            sys.stdout.write("\033[K" + m + "\r")

            basis_index = self._convert_to_basis_index(
                err_rnk, err_idx, samples.n_samples
            )
            self.error_list.append(error)
            basis_indicies.append(basis_index)

            # update iteration count
            iter += 1
        print("\n")

        self.matrix = np.asarray(self.matrix)
        self.greedypoints = [samples.samples[i] for i in basis_indicies]
        self.error_list = np.asarray(self.error_list)

        return self

        # if mpi.RANK_IS_MAIN:
        #    print("\nBasis generation complete!")
        #    if write_results:
        #        np.save("matrix", matrix)
        #        plot.errors(error_list)
        #        plot.basis(matrix)

    def write(self, h5file):
        """Save samples to file"""

        h5_group = h5file.create_group("reduced_basis")
        h5_group.create_dataset("matrix", data=self.matrix)
        h5_group.create_dataset("greedypoints", data=self.greedypoints)
        h5_group.create_dataset("error_list", data=self.error_list)

    @classmethod
    def from_file(cls, file_in):
        """Create a ROM instance from a file"""

        close_file = False
        if not isinstance(file_in, str):
            h5file = file_in
        else:
            h5file = h5py.File(file_in, "r")
            close_file = True

        matrix = np.array(h5file["reduced_basis/matrix"])
        greedypoints = [np.array(x) for x in h5file["reduced_basis/greedypoints"]]
        error_list = np.array(h5file["reduced_basis/error_list"])
        if close_file:
            h5file.close()
        return cls(matrix=matrix, greedypoints=greedypoints, error_list=error_list)

    def _add_next_model_to_basis(self, pc_matrix, my_ts, iter):
        # project training set on basis + get errors
        pc = misc.project_onto_basis(
            1.0, self.matrix, my_ts, iter - 1, self.model.model_dtype
        )
        pc_matrix.append(pc)

        projection_errors = list(
            1
            - np.einsum(
                "ij,ij->i", np.array(np.conjugate(pc_matrix)).T, np.array(pc_matrix).T
            )
        )

        # gather all errors (below is a list[ rank0_errors, rank1_errors...])
        all_rank_errors = mpi.COMM.gather(projection_errors, root=mpi.MAIN_RANK)

        # determine highest error
        if mpi.RANK_IS_MAIN:
            error_data = misc.get_highest_error(all_rank_errors)
            err_rank, err_idx, error = error_data
        else:
            error_data = None, None, None
        error_data = mpi.COMM.bcast(
            error_data, root=mpi.MAIN_RANK
        )  # share the error data with all nodes
        err_rank, err_idx, error = error_data

        # get model with the worst error
        worst_model = None
        if err_rank == mpi.MAIN_RANK:
            worst_model = my_ts[err_idx]  # no need to send
        elif mpi.RANK == err_rank:
            worst_model = my_ts[err_idx]
            mpi.COMM.send(worst_model, dest=mpi.MAIN_RANK)
        if worst_model is None and mpi.RANK_IS_MAIN:
            worst_model = mpi.COMM.recv(source=err_rank)

        # adding worst model to basis
        if mpi.RANK_IS_MAIN:
            # Gram-Schmidt to get the next basis and normalize
            self.matrix.append(misc.IMGS(self.matrix, worst_model, iter))

        # share the basis with ALL nodes
        matrix = mpi.COMM.bcast(self.matrix, root=mpi.MAIN_RANK)
        return matrix, pc_matrix, error_data

    def _convert_to_basis_index(self, rank_number, rank_idx, rank_count):
        ranks_till_err_rank = [i for i in range(rank_number)]
        idx_till_err_rank = np.sum([rank_count[i] for i in ranks_till_err_rank])
        return int(rank_idx + idx_till_err_rank)

    def _init_matrix(self, init_model):
        # init the baisis (matrix) with 1 model from the training set to start
        if mpi.RANK_IS_MAIN:
            self.matrix = [init_model]
        else:
            self.matrix = None
        # share the basis with ALL nodes
        self.matrix = mpi.COMM.bcast(self.matrix, root=mpi.MAIN_RANK)


class EmpiricalInterpolant(object):
    def __init__(self, B_matrix=None, nodes=None):
        """Initialise empirical interpolant"""

        self.B_matrix = B_matrix
        self.nodes = nodes

    def compute(self, reduced_basis):
        """Compute empirical interpolant"""

        # RB = RB[0 : len(RB)]
        if mpi.RANK_IS_MAIN:
            print("Computing empirical interpolant")
        eim = algorithms.StandardEIM(
            reduced_basis.matrix.shape[0], reduced_basis.matrix.shape[1]
        )
        eim.make(reduced_basis.matrix)
        domain = reduced_basis.model.domain
        self.nodes = domain[eim.indices]
        self.nodes, self.B_matrix = zip(*sorted(zip(self.nodes, eim.B)))

        return self

    def write(self, h5file):
        """Save empirical interpolant to file"""

        h5_group = h5file.create_group("empirical_interpolant")
        h5_group.create_dataset("B_matrix", data=self.B_matrix)
        h5_group.create_dataset("nodes", data=self.nodes)

    @classmethod
    def from_file(cls, file_in):
        """Create a ROM instance from a file"""

        close_file = False
        if not isinstance(file_in, str):
            h5file = file_in
        else:
            h5file = h5py.File(file_in, "r")
            close_file = True
        B_matrix = np.array(h5file["empirical_interpolant/B_matrix"])
        nodes = np.array(h5file["empirical_interpolant/nodes"])
        if close_file:
            h5file.close()
        return cls(B_matrix=B_matrix, nodes=nodes)
