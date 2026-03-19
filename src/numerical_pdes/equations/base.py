import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from numpy.typing import NDArray

from numerical_pdes.mesh import Mesh1D
from numerical_pdes.reconstruction import Reconstruction
from numerical_pdes.boundary_conditions import BoundaryCondition


class Equation(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def n_states(self) -> int:
        ...

    @abstractmethod
    def max_wave_speed(self, u: NDArray) -> float:
        """Return the maximum wave speed for CFL computation."""
        ...

    @abstractmethod
    def compute_numerical_flux(
        self,
        u: NDArray,
        mesh: Mesh1D,
        reconstruction: Reconstruction,
        bc: BoundaryCondition,
        n_ghost_cells: int,
    ) -> NDArray:
        """Compute the numerical flux at cell interfaces.

        Parameters
        ----------
        u : State array with ghost cells, shape (n_total, n_states).
        mesh : Computational mesh.
        reconstruction : Reconstruction strategy for interface values.
        bc : Boundary condition.
        n_ghost_cells : Number of ghost cells per side.

        Returns
        -------
        Flux array of shape (n_cells+1, n_states).
        """
        ...

    def compute_cfl_dt(self, u: NDArray, dx: float, cfl_number: float) -> float:
        speed = self.max_wave_speed(u)
        if speed < 1e-14:
            return dx
        return cfl_number * dx / speed

    def eigenvalues(self, u: NDArray) -> Tuple[float, float]:
        """Return (max_eigenvalue, min_eigenvalue) of the flux Jacobian."""
        return 0.0, 0.0

    def get_local_speed(
        self,
        u: NDArray,
        mesh: Mesh1D,
        reconstruction: Reconstruction,
        bc: BoundaryCondition,
        n_ghost_cells: int,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Central-upwind local speed computation.

        Returns (a_negative, a_positive, u_right, u_left) where
        a_negative/a_positive are the min/max local speeds and
        u_right/u_left are the reconstructed multi-component interface states.
        """
        n_cells = mesh.n_cells
        n_states = self.n_states

        a_negative = np.zeros((n_cells + 1, n_states))
        a_positive = np.zeros((n_cells + 1, n_states))
        u_right = np.zeros((n_cells + 1, n_states))
        u_left = np.zeros((n_cells + 1, n_states))

        for n in range(n_states):
            u_right[:, n], u_left[:, n] = reconstruction.reconstruct(
                u[:, n], mesh, n_ghost_cells, bc
            )

        for i in range(n_cells + 1):
            eig_max_l, eig_min_l = self.eigenvalues(u_left[i, :])
            eig_max_r, eig_min_r = self.eigenvalues(u_right[i, :])
            a_positive[i] = max(eig_max_l, eig_max_r, 0)
            a_negative[i] = min(eig_min_l, eig_min_r, 0)

        return a_negative, a_positive, u_right, u_left
