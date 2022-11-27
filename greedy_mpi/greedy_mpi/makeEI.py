import sys
import rompy
import numpy as np


RB = np.load("../RB_matrix.npy")

RB = RB[0:len(RB)]
eim = rompy.algorithms.StandardEIM(RB.shape[0], RB.shape[1])

eim.make(RB)

fseries = np.linspace(20, 2048, (2048-20)*256 +1)

fnodes = fseries[eim.indices]

fnodes, B = zip(*sorted(zip(fnodes, eim.B)))

np.save("B_matrix", B)
np.save("fnodes", fnodes)



