import numpy as np


def dot_product(model, weights, a, b):

    assert len(a) == len(b)
    if model.model_dtype == complex:
        return np.vdot(a * weights, b)
    else:
        return np.dot(a * weights, b)


def project_onto_basis(model, integration_weights, RB, my_ts, iter, dtype):

    pc = np.zeros(len(my_ts), dtype=dtype)
    for j in range(len(my_ts)):
        pc[j] = dot_product(model, integration_weights, RB[iter], my_ts[j])
    return pc


def MGS(model, RB, next_vec, iter):
    """what is this doing?"""
    dim_RB = iter
    for i in range(dim_RB):
        # --- ortho_basis = ortho_basis - L2_proj*basis; ---
        if model.model_dtype == complex:
            L2 = np.vdot(RB[i], next_vec)
        else:
            L2 = np.dot(RB[i], next_vec)
        next_vec -= RB[i] * L2
    if model.model_dtype == complex:
        norm = np.sqrt(np.vdot(next_vec, next_vec))
    else:
        norm = np.sqrt(np.dot(next_vec, next_vec))
    next_vec /= norm
    return next_vec, norm


def IMGS(model, RB, next_vec, iter):
    """what is this doing?"""
    ortho_condition = 0.5
    if model.model_dtype == complex:
        norm_prev = np.sqrt(np.vdot(next_vec, next_vec))
    else:
        norm_prev = np.sqrt(np.dot(next_vec, next_vec))
    flag = False
    while not flag:
        next_vec, norm_current = MGS(model, RB, next_vec, iter)
        next_vec *= norm_current
        if norm_current / norm_prev <= ortho_condition:
            norm_prev = norm_current
        else:
            flag = True
        if model.model_dtype == complex:
            norm_current = np.sqrt(np.vdot(next_vec, next_vec))
        else:
            norm_current = np.sqrt(np.dot(next_vec, next_vec))
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
