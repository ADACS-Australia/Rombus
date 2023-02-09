import sys
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import pylab as plt
from mpi4py import MPI
from tqdm.auto import tqdm

import rombus.misc as misc

MAIN_RANK = 0

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


class RombusModel(metaclass=ABCMeta):
    def __init__(self):
        self.init()

        # Ensure params is a list of strings
        assert bool(self.params) and all(isinstance(elem, str) for elem in self.params)

        # Ensure that model_dtype is a string
        assert type(self.model_dtype) == str

        # Create the named tuple that will be used for parameters
        self.params_dtype = namedtuple("params_dtype", self.params)

    def init(self):
        pass

    @property
    @abstractmethod  # make sure this is the inner-most decorator
    def model_dtype(self):
        pass

    @property
    @abstractmethod  # make sure this is the inner-most decorator
    def params(self):
        pass

    @abstractmethod  # make sure this is the inner-most decorator
    def init_domain(self):
        pass

    @abstractmethod  # make sure this is the inner-most decorator
    def compute(self, params: np.ndarray, domain) -> np.ndarray:
        pass


def generate_training_set(model, greedypoints: List[np.ndarray]) -> np.ndarray:
    """returns a list of waveforms (one for each row in 'greedypoints')"""

    domain = model.init_domain()

    my_ts = np.zeros(shape=(len(greedypoints), len(domain)), dtype=model.model_dtype)
    for ii, params_numpy in enumerate(
        tqdm(greedypoints, desc=f"Generating training set for rank {RANK}")
    ):
        params = model.params_dtype(
            **dict(zip(model.params, np.atleast_1d(params_numpy)))
        )
        h = model.compute(params, domain)
        my_ts[ii] = h / np.sqrt(np.vdot(h, h))
        # TODO: currently stored in RAM but does this need to be saved/cached on each
        #       compute node's scratch space?

    return my_ts


def divide_and_send_data_to_ranks(datafile: str) -> Tuple[List[np.ndarray], Dict]:
    # dividing greedypoints into chunks
    chunks = None
    chunk_counts = None
    if RANK == MAIN_RANK:
        if datafile.endswith(".npy"):
            greedypoints = np.load(datafile)
        elif datafile.endswith(".csv"):
            greedypoints = np.genfromtxt(datafile, delimiter=",")
        else:
            raise Exception

        chunks = [[] for _ in range(SIZE)]
        for i, chunk in enumerate(greedypoints):
            chunks[i % SIZE].append(chunk)
        chunk_counts = {i: len(chunks[i]) for i in range(len(chunks))}

    greedypoints = COMM.scatter(chunks, root=MAIN_RANK)
    chunk_counts = COMM.bcast(chunk_counts, root=MAIN_RANK)
    return greedypoints, chunk_counts


def init_basis_matrix(init_waveform):
    # init the baisis (RB_matrix) with 1 waveform from the training set to start
    if RANK == MAIN_RANK:
        RB_matrix = [init_waveform]
    else:
        RB_matrix = None
    RB_matrix = COMM.bcast(RB_matrix, root=MAIN_RANK)  # share the basis with ALL nodes
    return RB_matrix


def add_next_waveform_to_basis(RB_matrix, pc_matrix, my_ts, iter):
    # project training set on basis + get errors
    pc = misc.project_onto_basis(1.0, RB_matrix, my_ts, iter - 1, complex)
    pc_matrix.append(pc)
    # projection_errors = [
    #    1 - dot_product(1.0, np.array(pc_matrix).T[jj], np.array(pc_matrix).T[jj])
    #    for jj in range(len(np.array(pc_matrix).T))
    # ]
    # _l = len(np.array(pc_matrix).T)
    projection_errors = list(
        1
        - np.einsum(
            "ij,ij->i", np.array(np.conjugate(pc_matrix)).T, np.array(pc_matrix).T
        )
    )
    # gather all errors (below is a list[ rank0_errors, rank1_errors...])
    all_rank_errors = COMM.gather(projection_errors, root=MAIN_RANK)

    # determine  highest error
    if RANK == MAIN_RANK:
        error_data = misc.get_highest_error(all_rank_errors)
        err_rank, err_idx, error = error_data
    else:
        error_data = None, None, None
    error_data = COMM.bcast(
        error_data, root=MAIN_RANK
    )  # share the error data with all nodes
    err_rank, err_idx, error = error_data

    # get waveform with the worst error
    worst_waveform = None
    if err_rank == MAIN_RANK:
        worst_waveform = my_ts[err_idx]  # no need to send
    elif RANK == err_rank:
        worst_waveform = my_ts[err_idx]
        COMM.send(worst_waveform, dest=MAIN_RANK)
    if worst_waveform is None and RANK == MAIN_RANK:
        worst_waveform = COMM.recv(source=err_rank)

    # adding worst waveform to baisis
    if RANK == MAIN_RANK:
        # Gram-Schmidt to get the next basis and normalize
        RB_matrix.append(misc.IMGS(RB_matrix, worst_waveform, iter))

    # share the basis with ALL nodes
    RB_matrix = COMM.bcast(RB_matrix, root=MAIN_RANK)
    return RB_matrix, pc_matrix, error_data


def loop_log(iter, err_rnk, err_idx, err):
    m = f">>> Iter {iter:003}: err {err:.1E} (rank {err_rnk:002}@idx{err_idx:003})"
    sys.stdout.write("\033[K" + m + "\r")


def convert_to_basis_index(rank_number, rank_idx, rank_counts):
    ranks_till_err_rank = [i for i in range(rank_number)]
    idx_till_err_rank = np.sum([rank_counts[i] for i in ranks_till_err_rank])
    return int(rank_idx + idx_till_err_rank)


def plot_errors(err_list):
    plt.plot(err_list)
    plt.xlabel("# Basis elements")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("basis_error.png")


def plot_basis(rb_matrix):
    num_elements = len(rb_matrix)
    total_frames = 125
    h_in_one_frame = int(num_elements / total_frames)
    if h_in_one_frame < 1:
        h_in_one_frame = 1
    fig, ax = plt.subplots(total_frames, 1, figsize=(4.5, 2.5 * total_frames))
    for i in range(total_frames):
        start_i = int(i * h_in_one_frame)
        end_i = int(start_i + h_in_one_frame)
        for h_id in range(start_i, end_i):
            if end_i < num_elements:
                h = rb_matrix[h_id]
                ax[i].plot(h, color=f"C{h_id}", alpha=0.7)
        ax[i].set_title(f"Basis element {start_i:003}-{end_i:003}")
    plt.tight_layout()
    fig.savefig("basis.png")


def ROM(model, params: NamedTuple, domain, basis):
    _signal_at_nodes = model.compute(params, domain)
    return np.dot(_signal_at_nodes, basis)
