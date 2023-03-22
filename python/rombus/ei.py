import numpy as np

from typing import Self

import rombus._core.algorithms as algorithms
import rombus._core.mpi as mpi
import rombus._core.hdf5 as hdf5

from rombus.reduced_basis import ReducedBasis

DEFAULT_TOLERANCE = 1e-14
DEFAULT_REFINE_N_RANDOM = 100


class EmpiricalInterpolant(object):
    def __init__(
        self, B_matrix: np.ndarray = np.ndarray([]), nodes: np.ndarray = np.ndarray([])
    ):
        """Initialise empirical interpolant"""

        self.B_matrix = B_matrix
        self.nodes = nodes

    def compute(self, reduced_basis: ReducedBasis) -> Self:
        """Compute empirical interpolant"""

        # RB = RB[0 : len(RB)]
        if mpi.RANK_IS_MAIN:
            print("Computing empirical interpolant")
        print(reduced_basis.matrix_shape)
        eim = algorithms.StandardEIM(
            reduced_basis.matrix_shape[0], reduced_basis.matrix_shape[1]
        )
        eim.make(reduced_basis.matrix)
        domain = reduced_basis.model.domain
        self.nodes = domain[eim.indices]
        self.nodes, self.B_matrix = zip(*sorted(zip(self.nodes, eim.B)))

        return self

    def write(self, h5file: hdf5.File) -> None:
        """Save empirical interpolant to file"""

        h5_group = h5file.create_group("empirical_interpolant")
        h5_group.create_dataset("B_matrix", data=self.B_matrix)
        h5_group.create_dataset("nodes", data=self.nodes)

    @classmethod
    def from_file(cls, file_in: hdf5.FileOrFilename) -> Self:
        """Create a ROM instance from a file"""

        h5file, close_file = hdf5.ensure_open(file_in)
        B_matrix = np.array(h5file["empirical_interpolant/B_matrix"])
        nodes = np.array(h5file["empirical_interpolant/nodes"])
        if close_file:
            h5file.close()
        return cls(B_matrix=B_matrix, nodes=nodes)
