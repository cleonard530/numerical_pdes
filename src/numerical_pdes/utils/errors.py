import numpy as np
from numpy.typing import NDArray


def l1_error(u1: NDArray, u_ref: NDArray, is_relative: bool = True) -> float:
    err = np.linalg.norm(u_ref - u1, 1)
    if is_relative:
        err /= np.linalg.norm(u_ref, 1)
    return float(err)


def l2_error(u1: NDArray, u_ref: NDArray, is_relative: bool = True) -> float:
    err = np.linalg.norm(u_ref - u1, 2)
    if is_relative:
        err /= np.linalg.norm(u_ref, 2)
    return float(err)
