import numpy as np
import sys


def dot_product(weights, a, b):
    assert len(a) == len(b)
    return np.vdot(a * weights, b)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def project_onto_basis(integration_weights, e, h, pc, iter):
    # c = np.einsum('i,ji->j', integration_weights*e[iter].conjugate(), h)

    # projections += np.outer(c,e[iter])
    for j in range(len(h)):
        pc[j][iter] = dot_product(integration_weights, e[iter], h[j])
        # projections[j] += pc[j][iter]*e[iter]
    return pc


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def m1m2_to_Mc(m1, m2):
    """Chirp mass from m1, m2"""
    return (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def m1m2_to_nu(m1, m2):
    """Symmetric mass ratio from m1, m2"""
    return m1 * m2 / (m1 + m2) ** 2


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def m1m2_to_Mcnu(m1, m2):
    """Compute symmetric mass ratio and chirp mass from m1, m2"""
    return [m1m2_to_Mc(m1, m2), m1m2_to_nu(m1, m2)]


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def q_to_nu(q):
    """Convert mass ratio (which is >= 1) to symmetric mass ratio"""
    return q / (1.0 + q) ** 2.0


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def nu_to_q(nu):
    """Convert symmetric mass ratio to mass ratio (which is >= 1)"""
    return (1.0 + np.sqrt(1.0 - 4.0 * nu) - 2.0 * nu) / (2.0 * nu)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def Mq_to_m1m2(M, q):
    """Convert total mass, mass ratio pair to m1, m2"""
    m2 = M / (1.0 + q)
    m1 = M - m2
    return [m1, m2]


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def Mq_to_Mc(M, q):
    """Convert mass ratio, total mass pair to chirp mass"""
    return M * q_to_nu(q) ** (3.0 / 5.0)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def Mcq_to_M(Mc, q):
    """Convert mass ratio, chirp mass to total mass"""
    return Mc * q_to_nu(q) ** (-3.0 / 5.0)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def Mcnu_to_M(Mc, nu):
    """Convert chirp mass and symmetric mass ratio to total mass"""
    return Mc * nu ** (-3.0 / 5.0)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def Mnu_to_Mc(M, nu):
    """Convert total mass and symmetric mass ratio to chirp mass"""
    return M * nu ** (3.0 / 5.0)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def Mcnu_to_m1m2(Mc, nu):
    """Convert chirp mass, symmetric mass ratio pair to m1, m2"""
    q = nu_to_q(nu)
    M = Mcq_to_M(Mc, q)
    return Mq_to_m1m2(M, q)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def m1m2_to_delta(m1, m2):
    """Convert m1, m2 pair to relative mass difference [delta = (m1-m2)/(m1+m2)]"""
    return (m1 - m2) / (m1 + m2)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def delta_to_nu(delta):
    """Convert relative mass difference (delta) to symmetric mass ratio"""
    return (1.0 - delta**2) / 4.0


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def nu_to_delta(nu):
    """Convert symmetric mass ratio to relative mass difference delta"""
    return np.sqrt(1.0 - 4.0 * nu)


def MGS(RB, next_vec, iter):
    """what is this doing?"""
    dim_RB = iter
    for i in range(dim_RB):
        ## --- ortho_basis = ortho_basis - L2_proj*basis; --- ##
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
    return rank, idx, err
