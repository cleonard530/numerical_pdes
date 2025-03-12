import numpy as np
from numpy.typing import NDArray
from typing import List


def trig_function(x: NDArray,
                  a: NDArray | List | None = None,
                  b: NDArray | List | None = None,
                  k: int = 3,
                  period: float = 1.0,
                  t=0) -> NDArray:
    assert (a is None and b is None) or len(a) == len(b), f'a and be must both be None or the same value'
    if a is None and b is None:
        a = np.random.uniform(-1, 1, k)
        b = np.random.uniform(-1, 1, k)
    else:
        k = len(a)

    y = np.zeros(x.shape)
    for i in range(1, k + 1):
        y = y + (1 / i) * (a[i - 1] * np.cos(2*np.pi * i * (x+t) / period) +
                           b[i - 1] * np.sin(2*np.pi * i * (x+t) / period))
    return y


def piecewise_constant(n_cells: int, u_left: float | None = None, u_right: float | None = None):
    n2 = n_cells // 2
    if u_left is None:
        u_left = np.random.uniform(-1, 1)
    if u_right is None:
        u_right = np.random.uniform(-1, 1)

    u = np.zeros(n_cells)
    u[:n2] = u_left
    u[n2:] = u_right

    return u

