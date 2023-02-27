import sys
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
from mpi4py import MPI
from tqdm.auto import tqdm

import rombus.algorithms as algorithms
import rombus.misc as misc
import rombus.plot as plot

MAIN_RANK = 0

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

random = np.random.default_rng()


def generate_training_set(model, greedypoints: List[np.ndarray]) -> np.ndarray:
    """returns a list of models (one for each row in 'greedypoints')"""

    domain = model.init_domain()

    my_ts = np.zeros(shape=(len(greedypoints), len(domain)), dtype=model.model_dtype)
    for i, params_numpy in enumerate(
        tqdm(greedypoints, desc=f"Generating training set for rank {RANK}")
    ):
        params = model.params_dtype(
            **dict(zip(model.params, np.atleast_1d(params_numpy)))
        )
        model_i = model.compute(params, domain)
        my_ts[i] = model_i / np.sqrt(np.vdot(model_i, model_i))

    return my_ts


def read_divide_and_send_data_to_ranks(datafile: str) -> Tuple[List[np.ndarray], Dict]:
    # dividing greedypoints into chunks
    if RANK == MAIN_RANK:
        if datafile.endswith(".npy"):
            greedypoints = np.load(datafile)
        elif datafile.endswith(".csv"):
            greedypoints = np.genfromtxt(datafile, delimiter=",")
        else:
            raise Exception
    else:
        greedypoints = None

    return divide_and_send_data_to_ranks(greedypoints)


def divide_and_send_data_to_ranks(
    greedypoints: List[np.ndarray],
) -> Tuple[List[np.ndarray], Dict]:
    chunks = None
    chunk_counts = None
    if RANK == MAIN_RANK:
        chunks = [[] for _ in range(SIZE)]
        for i, chunk in enumerate(greedypoints):
            chunks[i % SIZE].append(chunk)
        chunk_counts = {i: len(chunks[i]) for i in range(len(chunks))}

    greedypoints = COMM.scatter(chunks, root=MAIN_RANK)
    chunk_counts = COMM.bcast(chunk_counts, root=MAIN_RANK)
    return greedypoints, chunk_counts


def init_basis_matrix(init_model):
    # init the baisis (RB_matrix) with 1 model from the training set to start
    if RANK == MAIN_RANK:
        RB_matrix = [init_model]
    else:
        RB_matrix = None
    RB_matrix = COMM.bcast(RB_matrix, root=MAIN_RANK)  # share the basis with ALL nodes
    return RB_matrix


def add_next_model_to_basis(RB_matrix, pc_matrix, my_ts, iter):
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

    # determine highest error
    if RANK == MAIN_RANK:
        error_data = misc.get_highest_error(all_rank_errors)
        err_rank, err_idx, error = error_data
    else:
        error_data = None, None, None
    error_data = COMM.bcast(
        error_data, root=MAIN_RANK
    )  # share the error data with all nodes
    err_rank, err_idx, error = error_data

    # get model with the worst error
    worst_model = None
    if err_rank == MAIN_RANK:
        worst_model = my_ts[err_idx]  # no need to send
    elif RANK == err_rank:
        worst_model = my_ts[err_idx]
        COMM.send(worst_model, dest=MAIN_RANK)
    if worst_model is None and RANK == MAIN_RANK:
        worst_model = COMM.recv(source=err_rank)

    # adding worst model to basis
    if RANK == MAIN_RANK:
        # Gram-Schmidt to get the next basis and normalize
        RB_matrix.append(misc.IMGS(RB_matrix, worst_model, iter))

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


def ROM(model, params: NamedTuple, domain, basis):
    _signal_at_nodes = model.compute(params, domain)
    return np.dot(_signal_at_nodes, basis)


def generate_random_samples(model, n_samples):
    samples = []
    for _ in range(n_samples):
        new_sample = np.ndarray(len(model.params), dtype=model.model_dtype)
        for i in range(len(model.params)):
            new_sample[i] = random.random() * 10.0
        samples.append(new_sample)
    return samples


def validate_and_refine_basis(
    model, RB_matrix, selected_greedy_points, tol, N_validations, write_results=True
):

    # generate validation set by randomly sampling the parameter space
    # need to think about how to do sampling, but this can be the same function
    # as the greedy points
    random_samples = generate_random_samples(model, N_validations)
    validation_samples, chunk_counts = divide_and_send_data_to_ranks(random_samples)

    my_vs = generate_training_set(model, validation_samples)

    n_added = 0
    RB_transpose = np.transpose(RB_matrix)
    for i, validation_sample in enumerate(validation_samples):
        proj_error = 1 - np.sum(np.dot(my_vs[i], RB_transpose))
        # proj_error = 1 - np.einsum( "ij,ij->i", np.array(np.conjugate(pc_matrix)).T, np.array(pc_matrix).T)
        print(f"params:{validation_sample} error={proj_error}")
        if proj_error > tol:
            selected_greedy_points.append(validation_sample)
            n_added = n_added + 1
    if RANK == MAIN_RANK:
        print(f"Number of samples added: {n_added}")

    # add the inaccurate points to the original selected greedy
    # points and remake the basis
    RB_updated, selected_greedy_points = make_reduced_basis(
        model, selected_greedy_points, chunk_counts, write_results=write_results
    )

    return RB_updated, selected_greedy_points


def refine(model, greedypoints, chunk_counts, tol=1e-4, N_validations=100):

    RB, selected_greedy_points = make_reduced_basis(
        model, greedypoints, chunk_counts, write_results=False
    )
    Refined_RB, _ = validate_and_refine_basis(
        model, RB, selected_greedy_points, tol, N_validations, write_results=False
    )

    np.save("RB_matrix", Refined_RB)
    EI = make_empirical_interpolant(model)

    # Write results
    np.save("EI", EI)

    return 0


def make_reduced_basis(model, greedypoints, chunk_counts, write_results=True):
    """Make reduced basis

    FILENAME_IN is the 'greedy points' numpy file to take as input
    """

    my_ts = generate_training_set(model, greedypoints)
    RB_matrix = init_basis_matrix(
        my_ts[0]
    )  # hardcoding 1st model to be used to start the basis

    error_list = []
    error = np.inf
    iter = 1
    basis_indicies = [0]  # we've used the 1st model already
    pc_matrix = []
    if RANK == MAIN_RANK:
        print("Filling basis with greedy-algorithm")
    while error > 1e-14:
        RB_matrix, pc_matrix, error_data = add_next_model_to_basis(
            RB_matrix, pc_matrix, my_ts, iter
        )
        err_rnk, err_idx, error = error_data

        # log and cache some data
        loop_log(iter, err_rnk, err_idx, error)

        basis_index = convert_to_basis_index(err_rnk, err_idx, chunk_counts)
        error_list.append(error)
        basis_indicies.append(basis_index)

        # update iteration count
        iter += 1

    if RANK == MAIN_RANK:
        print("\nBasis generation complete!")
        if write_results:
            np.save("RB_matrix", RB_matrix)
            plot.errors(error_list)
            plot.basis(RB_matrix)

    greedypoints_keep = [greedypoints[i] for i in basis_indicies]
    return RB_matrix, greedypoints_keep


def make_empirical_interpolant(model):
    """Make empirical interpolant"""

    RB = np.load("RB_matrix.npy")

    RB = RB[0 : len(RB)]
    eim = algorithms.StandardEIM(RB.shape[0], RB.shape[1])

    eim.make(RB)

    domain = model.init_domain()

    nodes = domain[eim.indices]

    nodes, B = zip(*sorted(zip(nodes, eim.B)))

    np.save("B_matrix", B)
    np.save("nodes", nodes)
