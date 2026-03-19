import numpy as np
from abc import ABC, abstractmethod
from typing import Callable
from numpy.typing import NDArray

from numerical_pdes.boundary_conditions import BoundaryCondition


class TimeIntegrator(ABC):
    @abstractmethod
    def step(
        self,
        u: NDArray,
        dt: float,
        rhs_fn: Callable[[NDArray], NDArray],
        bc: BoundaryCondition,
        n_ghost_cells: int,
    ) -> NDArray:
        """Advance the solution by one time step of size dt.

        Parameters
        ----------
        u : Full state array with ghost cells, shape (n_total, n_states).
        dt : Time step size.
        rhs_fn : Callable that computes the spatial RHS on interior cells.
        bc : Boundary condition to apply after each stage.
        n_ghost_cells : Number of ghost cells on each side.

        Returns
        -------
        Updated state array (same shape as u).
        """
        ...


class ForwardEuler(TimeIntegrator):
    def step(self, u, dt, rhs_fn, bc, n_ghost_cells):
        gc = n_ghost_cells
        u[gc:-gc, :] += dt * rhs_fn(u)
        bc.apply(u, n_ghost_cells)
        return u


class SSPRK3(TimeIntegrator):
    """Third-order strong-stability-preserving Runge-Kutta."""

    def step(self, u, dt, rhs_fn, bc, n_ghost_cells):
        gc = n_ghost_cells
        u1 = np.zeros_like(u)
        u2 = np.zeros_like(u)

        rhs = rhs_fn(u)
        u1[gc:-gc, :] = u[gc:-gc, :] + dt * rhs
        bc.apply(u1, n_ghost_cells)

        rhs = rhs_fn(u1)
        u2[gc:-gc, :] = 0.75 * u[gc:-gc, :] + 0.25 * (u1[gc:-gc, :] + dt * rhs)
        bc.apply(u2, n_ghost_cells)

        rhs = rhs_fn(u2)
        u[gc:-gc, :] = (1 / 3) * u[gc:-gc, :] + (2 / 3) * (u2[gc:-gc, :] + dt * rhs)
        bc.apply(u, n_ghost_cells)

        return u
