import sys
from greedy_mpi.rompy import *
import numpy as np


RB = np.load("RB_matrix_dimensionless_physical_dimension.npy")

RB = RB
eim = algorithms.StandardEIM(RB.shape[0], RB.shape[1])

eim.make(RB)

#fseries = np.linspace(0, 1000*100-1, 1000*100)

#fnodes = fseries[eim.indices]

indices, B = zip(*sorted(zip(eim.indices, eim.B)))

np.save("B_matrix_dimensionless_physical_dimension", B)
np.save("xy_ravel_indices_dimensionless_physical_dimension", indices)



