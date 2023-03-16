import numpy as np


def dot_product(weights, a, b):

    assert len(a) == len(b)
    return np.vdot(a * weights, b)


def project_onto_basis(integration_weights, e, my_ts, iter, dtype):

    # c = np.einsum('i,ji->j', integration_weights*e[iter].conjugate(), h)
    pc = np.zeros(len(my_ts), dtype=dtype)
    # projections += np.outer(c,e[iter])
    for j in range(len(my_ts)):
        pc[j] = dot_product(integration_weights, e[iter], my_ts[j])
        # projections[j] += pc[j][iter]*e[iter]
    return pc


def MGS(RB, next_vec, iter):
    """what is this doing?"""
    dim_RB = iter
    for i in range(dim_RB):
        # --- ortho_basis = ortho_basis - L2_proj*basis; ---
        L2 = np.vdot(RB[i], next_vec)
        next_vec -= RB[i] * L2
    norm = np.sqrt(np.vdot(next_vec, next_vec))
    next_vec /= norm
    return next_vec, norm


def IMGS(RB, next_vec, iter):
    """what is this doing?"""
    ortho_condition = 0.5
    norm_prev = np.sqrt(np.vdot(next_vec, next_vec))
    flag = False
    while not flag:
        next_vec, norm_current = MGS(RB, next_vec, iter)
        next_vec *= norm_current
        if norm_current / norm_prev <= ortho_condition:
            norm_prev = norm_current
        else:
            flag = True
        norm_current = np.sqrt(np.vdot(next_vec, next_vec))
        next_vec /= norm_current
    # RB[iter] = next_vec  # np.vstack((RB, next_vec))
    # return RB[iter]
    return next_vec


def get_highest_error(error_list):
    rank, idx, err = -np.inf, -np.inf, -np.inf
    for rank_id, rank_errors in enumerate(error_list):
        max_rank_err = max(rank_errors)
        if max_rank_err > err:
            err = max_rank_err
            idx = rank_errors.index(err)
            rank = rank_id
    return rank, idx, np.float64(err.real)
