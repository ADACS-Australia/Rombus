import sys
from greedy_mpi.rompy import *
import numpy as np


RB = np.load("RB_matrix.npy")

RB = RB[0:len(RB)]
eim = algorithms.StandardEIM(RB.shape[0], RB.shape[1])

eim.make(RB)

wavelengths = np.linspace(0,len(RB[0])-1, len(RB[0]))

nodes = wavelengths[eim.indices]

nodes, B = zip(*sorted(zip(nodes, eim.B)))

np.save("B_matrix", B)
np.save("nodes", nodes)



