from rombus._core.log import log

try:
    from mpi4py import MPI
except ImportError:
    log.comment("MPI not detected.  Running without it.")
    active = False
    COMM = None
    SIZE = 1
    RANK = 0
else:
    active = True
    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()
    if SIZE == 1:
        log.comment(f"MPI detected.  Running with {SIZE} core.")
    else:
        log.comment(f"MPI detected.  Running with {SIZE} cores.")
import numpy as np

MAIN_RANK = 0

RANK_IS_MAIN = RANK == MAIN_RANK


def allreduce_SUM(data: np.ndarray) -> np.ndarray:
    """MPI wrapper for the allreduce sum operation.

    Args:
        data: numpy data to operate on

    Returns:
        Sum of data on all ranks
    """
    if active:
        return COMM.allreduce(data, op=MPI.SUM)
    else:
        return np.sum(data)


def bcast(data: np.ndarray, root: int = MAIN_RANK) -> np.ndarray:
    """MPI wrapper for the bcast operation

    Args:
        data: data to send
        root: rank to send it from

    Returns:
        The broadcasted result on all ranks
    """
    if active:
        return COMM.bcast(data, root=root)
    else:
        return data


def gather(data: np.ndarray, root: int = MAIN_RANK) -> np.ndarray:
    """MPI wrapper for the bcast operation

    Args:
        data: data to send
        root: rank to send it from

    Returns:
        Gathered data on the root rank; None otherwise
    """
    if active:
        return COMM.gather(data, root=root)
    else:
        return [data]


def Send(data: np.ndarray, dest: int = MAIN_RANK) -> None:
    """MPI wrapper for the send operation

    Args:
        data: data to send
        root: rank to send it from

    Returns:
        None
    """
    if active:
        COMM.Send(data, dest=dest)
    else:
        raise NotImplementedError


def Recv(data: np.ndarray, source: int = MAIN_RANK) -> None:
    """MPI wrapper for the receive operation

    Args:
        data: data to receive
        source: rank to receive it from

    Returns:
        Sent data from the source rank; None otherwise
    """
    if active:
        COMM.Recv(data, source=source)
    else:
        raise NotImplementedError


def Scatter(data: np.ndarray, root: int = MAIN_RANK) -> np.ndarray:
    """MPI wrapper for the scatter operation

    Args:
        data: data to receive
        source: rank to receive it from

    Returns:
        Sent data from the source rank; None otherwise
    """
    if active:
        return COMM.scatter(data, root=root)
    else:
        return data[0]
