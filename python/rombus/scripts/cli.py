"""
run this file with:
mpirun -n 3 python make_reduced_basis.py

"""

import click
import numpy as np
import lalsimulation
from rombus.misc import *
import rombus as rb
import lal
import sys
from mpi4py import MPI
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import pylab as plt
from dataclasses import dataclass, field
from typing import Dict, Protocol

MAIN_RANK = 0

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

class IsDataclass(Protocol):
    # Checking for this attribute is currently the most reliable way to 
    # ascertain that something is a dataclass
    __dataclass_fields__: Dict

############ START OF DOMAIN-SPECIFIC CODE ############ 

model_dtype = 'complex'

def init_cache():
    @dataclass
    class ModelCache(object):
        fmin: float
        fmax: float
        deltaF: float
        nf: int
        fseries: np.ndarray
        fmin_index: float
        WFdict: Dict
    fmin = 20
    fmax = 1024
    deltaF = 1.0 / 4.0
    nf = int((fmax-fmin)/deltaF)+1
    fseries = np.linspace(fmin, fmax, nf)
    fmin_index = int(fmin / deltaF)
    WFdict = lal.CreateDict()
    return ModelCache(
        fmin = fmin,
        fmax = fmax,
        deltaF = deltaF,
        nf = nf,
        fseries = fseries,
        fmin_index = fmin_index,
        WFdict = WFdict
        )

def init_domain(cache):
    return cache.fseries

def compute_model(params: np.array, domain, cache: IsDataclass) -> np.array:
    m1 = params[0]
    m2 = params[1]
    chi1L = params[2]
    chi2L = params[3]
    chip = params[4]
    thetaJ = params[5]
    alpha = params[6]
    l1 =0
    l2 =0

    m1 *= lal.lal.MSUN_SI
    m2 *= lal.lal.MSUN_SI

    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(cache.WFdict, l1)
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(cache.WFdict, l2)

    if not np.array_equiv(domain,cache.fseries):
        h = lalsimulation.SimIMRPhenomPFrequencySequence(
            domain,
            chi1L,
            chi2L,
            chip,
            thetaJ,
            m1,
            m2,
            1e6 * lal.lal.PC_SI * 100,
            alpha,
            0,
            40,
            lalsimulation.IMRPhenomPv2NRTidal_V,
            lalsimulation.NRTidalv2_V,
            cache.WFdict,
        )
        h = h[0].data.data
    else:
        h = lalsimulation.SimIMRPhenomP(
            chi1L,
            chi2L,
            chip,
            thetaJ,
            m1,
            m2,
            1e6 * lal.lal.PC_SI * 100,
            alpha,
            0,
            cache.deltaF,
            cache.fmin,
            cache.fmax,
            40,
            lalsimulation.IMRPhenomPv2NRTidal_V,
            lalsimulation.NRTidalv2_V,
            cache.WFdict,
        )
        h = h[0].data.data[cache.fmin_index: len(h[0].data.data)]
        if len(h) < cache.nf:
            h = np.append(h, np.zeros(cache.nf - len(h), dtype=complex))

    return h

############ END OF DOMAIN-SPECIFIC CODE ############ 

def generate_training_set(greedypoints: List[np.array]) -> List[np.array]:
    """returns a list of waveforms (one for each row in 'greedypoints')"""

    cache = init_cache()
    domain = init_domain(cache)

    my_ts = np.zeros(shape=(len(greedypoints), len(domain)), dtype=model_dtype)
    for ii, params in enumerate(tqdm(greedypoints, desc=f"Generating training set for rank {RANK}")):
        h = compute_model(params, domain, cache)
        my_ts[ii] = h / np.sqrt(np.vdot(h, h))
        # TODO: currently stored in RAM but does this need to be saved/cached on each compute node's scratch space?

    return my_ts

def divide_and_send_data_to_ranks(datafile: str) -> Tuple[List[np.array], Dict]:
    # dividing greedypoints into chunks
    chunks = None
    chunk_counts = None
    if RANK == MAIN_RANK:
        greedypoints = np.load(datafile)
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
    pc = project_onto_basis(1.0, RB_matrix, my_ts, iter - 1, complex)
    pc_matrix.append(pc)
    #projection_errors = [
    #    1 - dot_product(1.0, np.array(pc_matrix).T[jj], np.array(pc_matrix).T[jj])
    #    for jj in range(len(np.array(pc_matrix).T))
    #]
    #_l = len(np.array(pc_matrix).T)
    projection_errors = list(1-np.einsum('ij,ij->i', np.array(np.conjugate(pc_matrix)).T,np.array(pc_matrix).T ))
    # gather all errors (below is a list[ rank0_errors, rank1_errors...])
    all_rank_errors = COMM.gather(projection_errors, root=MAIN_RANK)

    # determine  highest error
    if RANK == MAIN_RANK:
        error_data = get_highest_error(all_rank_errors)
        err_rank, err_idx, error = error_data
    else:
        error_data = None, None, None
    error_data = COMM.bcast( error_data, root=MAIN_RANK)  # share the error data with all nodes
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
        RB_matrix.append(IMGS(RB_matrix, worst_waveform, iter))

    # share the basis with ALL nodes
    RB_matrix = COMM.bcast(RB_matrix, root=MAIN_RANK)
    return RB_matrix, pc_matrix, error_data


def loop_log(iter, err_rnk, err_idx, err):
    m = f">>> Iter {iter:003}: err {err:.1E} (rank {err_rnk:002}@idx{err_idx:003})"
    sys.stdout.write('\033[K' + m + '\r')


def convert_to_basis_index(rank_number, rank_idx, rank_counts):
    ranks_till_err_rank = [i for i in range(rank_number)]
    idx_till_err_rank = np.sum([rank_counts[i] for i in ranks_till_err_rank])
    return int(rank_idx + idx_till_err_rank)


def plot_errors(err_list):
    plt.plot(err_list)
    plt.xlabel("# Basis elements")
    plt.ylabel("Error")
    plt.yscale('log')
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

def ROM(params, domain, cache, basis):
    _signal_at_nodes = compute_model(params, domain, cache)
    return np.dot(_signal_at_nodes, basis)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group()
@click.pass_context
def cli(ctx):
    """Perform greedy algorythm operations with Rombus

    """

    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument('filename_in', type=click.Path(exists=True))
@click.pass_context
def make_reduced_basis(ctx,filename_in):
    """Make reduced basis

    FILENAME_IN is the 'greedy points' numpy file to take as input
    """

    greedypoints, chunk_counts = divide_and_send_data_to_ranks(filename_in)
    my_ts = generate_training_set(greedypoints)
    RB_matrix = init_basis_matrix(my_ts[0])  # hardcoding 1st waveform to be used to start the basis

    error_list = []
    error = np.inf
    iter = 1
    basis_indicies = [0]  # we've used the 1st waveform already
    pc_matrix = []
    if RANK == MAIN_RANK:
        print("Filling basis with greedy-algorithm")
    while error > 1e-14:
        RB_matrix, pc_matrix, error_data = add_next_waveform_to_basis(
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
        np.save("RB_matrix", RB_matrix)
        greedypoints = np.load(filename_in)
        np.save("GreedyPoints", greedypoints[basis_indicies])
        plot_errors(error_list)
        plot_basis(RB_matrix)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.pass_context
def make_empirical_interpolant(ctx):
    """Make empirical interpolant
    """

    RB = np.load("RB_matrix.npy")
    
    RB = RB[0:len(RB)]
    eim = rb.algorithms.StandardEIM(RB.shape[0], RB.shape[1])
    
    eim.make(RB)
    
    cache = init_cache()
    domain = init_domain(cache)
    
    fnodes = domain[eim.indices]
    
    fnodes, B = zip(*sorted(zip(fnodes, eim.B)))
    
    np.save("B_matrix", B)
    np.save("fnodes", fnodes)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.pass_context
def compare_rom_to_true(ctx):
    """Compare computed ROM to input model

    FILENAME_IN is the 'greedy points' numpy file to take as input
    """

    basis = np.load("B_matrix.npy")
    fnodes = np.load("fnodes.npy")

    m_min = 20
    m_max = 30

    m1 = np.random.uniform(low=m_min, high=m_max)
    m2  = np.random.uniform(low=m_min, high=m_max)
    chi1L = np.random.uniform(low=0, high=0.8)
    chi2L  = np.random.uniform(low=0, high=0.8)
    chip  = np.random.uniform(low=0, high=0.8)
    thetaJ = np.random.uniform(low=0, high=np.pi)
    alpha = np.random.uniform(low=0, high=np.pi)

    params = np.array([m1, m2, chi1L, chi2L, chip, thetaJ, alpha])

    cache = init_cache()
    domain = init_domain(cache)

    h_full = compute_model(params, domain, cache)
    h_nodes = compute_model(params, fnodes, cache)
    h_rom = ROM(params, fnodes, cache, basis)

    np.save("ROM_diff", np.subtract(h_rom,h_full))

    plt.semilogx(domain, h_rom, label='ROM', alpha=0.5, linestyle='--')
    plt.semilogx(domain, h_full, label='Full model', alpha=0.5)
    plt.scatter(fnodes, h_nodes, s=1)
    plt.legend()
    plt.savefig("comparison.pdf", bbox_inches='tight')

if __name__ == '__main__':
    cli(obj={})
    sys.exit()
