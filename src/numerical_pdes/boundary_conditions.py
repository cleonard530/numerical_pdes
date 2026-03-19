from abc import ABC, abstractmethod
from numpy.typing import NDArray


class BoundaryCondition(ABC):
    @abstractmethod
    def apply(self, u: NDArray, n_ghost_cells: int) -> None:
        """Fill volume ghost cells in-place."""
        ...

    @abstractmethod
    def apply_interface(self, u_right: NDArray, u_left: NDArray, n_cells: int) -> None:
        """Fill interface boundary values in-place after reconstruction."""
        ...


class PeriodicBC(BoundaryCondition):
    def apply(self, u: NDArray, n_ghost_cells: int) -> None:
        n_interior = u.shape[0] - 2 * n_ghost_cells
        for i in range(n_ghost_cells):
            u[i, :] = u[i + n_interior, :]
            u[-1 - i, :] = u[-1 - i - n_interior, :]

    def apply_interface(self, u_right: NDArray, u_left: NDArray, n_cells: int) -> None:
        u_left[0] = u_left[n_cells]
        u_right[n_cells] = u_right[0]


class WallBC(BoundaryCondition):
    def apply(self, u: NDArray, n_ghost_cells: int) -> None:
        for i in range(n_ghost_cells):
            u[i, :] = u[2 * n_ghost_cells - 1 - i, :]
            u[-1, :] = u[-(2 * n_ghost_cells) + i, :]

    def apply_interface(self, u_right: NDArray, u_left: NDArray, n_cells: int) -> None:
        u_left[0] = u_right[0]
        u_right[n_cells] = u_left[n_cells]
