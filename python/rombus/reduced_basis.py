import sys

import numpy as np

from typing import List, Self, Tuple

import rombus._core.mpi as mpi
import rombus._core.hdf5 as hdf5
import rombus.exceptions as exceptions
from rombus.samples import Samples
from rombus.model import RombusModel, RombusModelType

DEFAULT_TOLERANCE = 1e-14
DEFAULT_REFINE_N_RANDOM = 100

def _get_highest_error(error_list):
    rank, idx, err = -np.inf, -np.inf, -np.inf
    for rank_id, rank_errors in enumerate(error_list):
        max_rank_err = max(rank_errors)
        if max_rank_err > err:
            err = max_rank_err
            idx = rank_errors.index(err)
            rank = rank_id
    return rank, idx, np.float64(err.real)

def _dot_product(weights, a, b):

    assert len(a) == len(b)
    return np.vdot(a * weights, b)

class ReducedBasis(object):
    def __init__(
        self,
        matrix: List[np.ndarray] = [],
        greedypoints: List[np.ndarray] = [],
        error_list: List[float] = [],
        matrix_shape: List[int] = [0, 0],
    ):
        try:
            self.matrix = matrix
            self.greedypoints = greedypoints
            self.error_list = error_list
            self._set_matrix_shape()
        except exceptions.RombusException as e:
            e.handle_exception()

    def _set_matrix_shape(self) -> None:
        mtx_len = len(self.matrix)
        self.matrix_shape = [mtx_len]
        if mtx_len > 0:
            mtx_dim = self.matrix[-1].shape
            for dim in mtx_dim:
                self.matrix_shape.append(dim)
        else:
            self.matrix_shape.append(0)

    def compute(
        self, model: RombusModelType, samples: Samples, tol: float = DEFAULT_TOLERANCE
    ) -> Self:

        self.model: RombusModel = RombusModel.load(model)

        # Compute the model for each given sample
        my_ts: np.ndarray = self.model.generate_model_set(samples)

        if mpi.RANK_IS_MAIN:
            print("Filling basis with greedy-algorithm")

        # hardcoding 1st model to be used to start the basis
        self._init_matrix(my_ts[0])
        basis_indicies = [0]

        # NOTE: NEED TO FIX; THE FIRST INDEX IS BEING ADDED TWICE HERE!
        error = np.inf
        iter = 1
        pc_matrix: List[np.ndarray] = []
        while error > tol:
            self.matrix, pc_matrix, error_data = self._add_next_model_to_basis(
                pc_matrix, my_ts, iter
            )
            self.matrix_shape[0] = len(self.matrix)
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

        self.greedypoints = [samples.samples[i] for i in basis_indicies]

        return self

    def write(self, h5file: hdf5.File) -> None:
        """Save samples to file"""

        h5_group = h5file.create_group("reduced_basis")
        h5_group.create_dataset("matrix", data=self.matrix)
        h5_group.create_dataset("greedypoints", data=self.greedypoints)
        h5_group.create_dataset("error_list", data=self.error_list)

    @classmethod
    def from_file(cls, file_in: hdf5.FileOrFilename) -> Self:
        """Create a ROM instance from a file"""

        h5file, close_file = hdf5.ensure_open(file_in)

        matrix = [np.array(x) for x in h5file["reduced_basis/matrix"]]
        greedypoints = [np.array(x) for x in h5file["reduced_basis/greedypoints"]]
        error_list = [x for x in h5file["reduced_basis/error_list"]]

        if close_file:
            h5file.close()

        return cls(matrix=matrix, greedypoints=greedypoints, error_list=error_list)

    def _add_next_model_to_basis(
        self, pc_matrix: List[np.ndarray], my_ts: np.ndarray, iter: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Tuple[int, int, int]]:
        # project training set on basis + get errors
        pc = self._project_onto_basis( 1.0, my_ts, iter - 1)
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
            error_data = _get_highest_error(all_rank_errors)
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
            self.matrix.append(self._IMGS(worst_model, iter))

        # share the basis with ALL nodes
        matrix = mpi.COMM.bcast(self.matrix, root=mpi.MAIN_RANK)
        return matrix, pc_matrix, error_data

    def _IMGS(self, next_vec, iter):
        """what is this doing?"""
        ortho_condition = 0.5
        norm_prev = np.sqrt(np.vdot(next_vec, next_vec))
        flag = False
        while not flag:
            next_vec, norm_current = self._MGS( next_vec, iter)
            next_vec *= norm_current
            if norm_current / norm_prev <= ortho_condition:
                norm_prev = norm_current
            else:
                flag = True
            norm_current = np.sqrt(np.vdot(next_vec, next_vec))
            next_vec /= norm_current
        return next_vec

    def _MGS(self, next_vec, iter):
        """what is this doing?"""
        dim_RB = iter
        for i in range(dim_RB):
            # --- ortho_basis = ortho_basis - L2_proj*basis; ---
            L2 = np.vdot(self.matrix[i], next_vec)
            next_vec -= self.matrix[i] * L2
        norm = np.sqrt(np.vdot(next_vec, next_vec))
        next_vec /= norm
        return next_vec, norm

    def _project_onto_basis(self, integration_weights,  my_ts, iter):

        pc = np.zeros(len(my_ts), dtype=self.model.model_dtype)
        for j in range(len(my_ts)):
            pc[j] = _dot_product(integration_weights, self.matrix[iter], my_ts[j])
        return pc

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
        self._set_matrix_shape()
