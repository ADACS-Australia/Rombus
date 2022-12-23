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
from rombus.importer import ImportFromStringError, import_from_string
import rombus.core as core

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group()
@click.argument('model', type=str)
@click.pass_context
def cli(ctx, model):
    """Perform greedy algorythm operations with Rombus

    """

    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)

    # Import user code defining the model we are working with
    model_class = import_from_string(model)
    model_instance = model_class()
    ctx.obj = model_instance

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument('filename_in', type=click.Path(exists=True))
@click.pass_context
def make_reduced_basis(ctx,filename_in):
    """Make reduced basis

    FILENAME_IN is the 'greedy points' numpy file to take as input
    """

    model = ctx.obj

    greedypoints, chunk_counts = core.divide_and_send_data_to_ranks(filename_in)
    my_ts = core.generate_training_set(model, greedypoints)
    RB_matrix = core.init_basis_matrix(my_ts[0])  # hardcoding 1st waveform to be used to start the basis

    error_list = []
    error = np.inf
    iter = 1
    basis_indicies = [0]  # we've used the 1st waveform already
    pc_matrix = []
    if core.RANK == core.MAIN_RANK:
        print("Filling basis with greedy-algorithm")
    while error > 1e-14:
        RB_matrix, pc_matrix, error_data = core.add_next_waveform_to_basis(
            RB_matrix, pc_matrix, my_ts, iter
        )
        err_rnk, err_idx, error = error_data

        # log and cache some data
        core.loop_log(iter, err_rnk, err_idx, error)

        basis_index = core.convert_to_basis_index(err_rnk, err_idx, chunk_counts)
        error_list.append(error)
        basis_indicies.append(basis_index)

        # update iteration count
        iter += 1

    if core.RANK == core.MAIN_RANK:
        print("\nBasis generation complete!")
        np.save("RB_matrix", RB_matrix)
        greedypoints = np.load(filename_in)
        np.save("GreedyPoints", greedypoints[basis_indicies])
        core.plot_errors(error_list)
        core.plot_basis(RB_matrix)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.pass_context
def make_empirical_interpolant(ctx):
    """Make empirical interpolant
    """

    model = ctx.obj

    RB = np.load("RB_matrix.npy")
    
    RB = RB[0:len(RB)]
    eim = rb.algorithms.StandardEIM(RB.shape[0], RB.shape[1])
    
    eim.make(RB)
    
    domain = model.init_domain()
    
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

    model = ctx.obj

    basis = np.load("B_matrix.npy")
    fnodes = np.load("fnodes.npy")

    ########## THIS DOMAIN-SPECIFIC CODE HAS TO GO AT SOME POINT ########## 
    ########## THE REAL PROBLEM HERE IS WITH HOW GREEDYPOINTS    ########## 
    ########## ARE HANDLED; MUST GENERALISE                      ########## 
    m_min = 20
    m_max = 30

    m1 = np.random.uniform(low=m_min, high=m_max)
    m2  = np.random.uniform(low=m_min, high=m_max)
    chi1L = np.random.uniform(low=0, high=0.8)
    chi2L  = np.random.uniform(low=0, high=0.8)
    chip  = np.random.uniform(low=0, high=0.8)
    thetaJ = np.random.uniform(low=0, high=np.pi)
    alpha = np.random.uniform(low=0, high=np.pi)
    ####################################################################### 

    params = np.array([m1, m2, chi1L, chi2L, chip, thetaJ, alpha])

    domain = model.init_domain()

    h_full = model.compute(params, domain)
    h_nodes = model.compute(params, fnodes)
    h_rom = core.ROM(model, params, fnodes, basis)

    np.save("ROM_diff", np.subtract(h_rom,h_full))

    plt.semilogx(domain, h_rom, label='ROM', alpha=0.5, linestyle='--')
    plt.semilogx(domain, h_full, label='Full model', alpha=0.5)
    plt.scatter(fnodes, h_nodes, s=1)
    plt.legend()
    plt.savefig("comparison.pdf", bbox_inches='tight')

if __name__ == '__main__':
    cli(obj={})
    sys.exit()
