import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from numpy.typing import NDArray

from numerical_pdes.mesh import Mesh1D
from numerical_pdes.boundary_conditions import BoundaryCondition


class Reconstruction(ABC):
    @property
    @abstractmethod
    def required_ghost_cells(self) -> int:
        ...

    @abstractmethod
    def reconstruct(
        self,
        u: NDArray,
        mesh: Mesh1D,
        n_ghost_cells: int,
        bc: BoundaryCondition,
    ) -> Tuple[NDArray, NDArray]:
        """Reconstruct interface values from cell averages.

        Parameters
        ----------
        u : 1-D array of shape (n_cells + 2*n_ghost_cells,) for one state component.
        mesh : The computational mesh.
        n_ghost_cells : Number of ghost cells on each side.
        bc : Boundary condition (used for interface boundary values).

        Returns
        -------
        (u_right, u_left) each of shape (n_cells+1,).
            u_right[i] = limit from the right at interface i.
            u_left[i]  = limit from the left  at interface i.
        """
        ...


class ConstantReconstruction(Reconstruction):
    @property
    def required_ghost_cells(self) -> int:
        return 1

    def reconstruct(self, u, mesh, n_ghost_cells, bc):
        n_cells = mesh.n_cells
        u_right = np.zeros(n_cells + 1)
        u_left = np.zeros(n_cells + 1)

        offset = n_ghost_cells - 1
        for i in range(n_cells + 1):
            u_left[i] = u[offset + i]
            u_right[i] = u[offset + i + 1]

        bc.apply_interface(u_right, u_left, n_cells)
        return u_right, u_left


class MinmodLinearReconstruction(Reconstruction):
    def __init__(self, theta: float = 1.0):
        self.theta = theta

    @property
    def required_ghost_cells(self) -> int:
        return 1

    def reconstruct(self, u, mesh, n_ghost_cells, bc):
        n_cells = mesh.n_cells
        dx = mesh.dx
        u_right = np.zeros(n_cells + 1)
        u_left = np.zeros(n_cells + 1)

        for i in range(n_cells):
            center = n_ghost_cells + i
            du = self._du_dx(u[center - 1], u[center], u[center + 1], dx)
            u_left[i + 1] = u[center] + du * (dx / 2)
            u_right[i] = u[center] - du * (dx / 2)

        bc.apply_interface(u_right, u_left, n_cells)
        return u_right, u_left

    def _du_dx(self, um1: float, u: float, up1: float, dx: float) -> float:
        forward = self.theta * (up1 - u)
        backward = self.theta * (u - um1)
        central = (up1 - um1) / 2
        return _minmod(forward, backward, central) / dx


def _minmod(a: float, b: float, c: float) -> float:
    if a > 0 and b > 0:
        return min(a, b, c)
    elif a < 0 and b < 0:
        return max(a, b, c)
    return 0.0
