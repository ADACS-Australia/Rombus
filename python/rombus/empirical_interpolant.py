import h5py
import numpy as np

import rombus._core.algorithms as algorithms
import rombus._core.mpi as mpi

DEFAULT_TOLERANCE = 1e-14
DEFAULT_REFINE_N_RANDOM = 100


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
