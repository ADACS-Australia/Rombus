import numpy as np
import pylab as plt

import rombus.core as core


def errors(err_list):
    plt.plot(err_list)
    plt.xlabel("# Basis elements")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("basis_error.png")


def basis(rb_matrix):
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


def compare_rom_to_true(model, model_params_in):
    """Compare computed ROM to input model"""

    basis = np.load("B_matrix.npy")
    fnodes = np.load("fnodes.npy")

    model_params = model.params_dtype(**model_params_in)

    domain = model.init_domain()

    h_full = model.compute(model_params, domain)
    h_nodes = model.compute(model_params, fnodes)
    h_rom = core.ROM(model, model_params, fnodes, basis)

    np.save("ROM_diff", np.subtract(h_rom, h_full))

    plt.semilogx(domain, h_rom, label="ROM", alpha=0.5, linestyle="--")
    plt.semilogx(domain, h_full, label="Full model", alpha=0.5)
    plt.scatter(fnodes, h_nodes, s=1)
    plt.legend()
    plt.savefig("comparison.pdf", bbox_inches="tight")
