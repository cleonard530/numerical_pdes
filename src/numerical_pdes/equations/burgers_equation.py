import numpy as np
from typing import Tuple
from numpy.typing import NDArray

from numerical_pdes.equations.base import Equation
from numerical_pdes.mesh import Mesh1D
from numerical_pdes.reconstruction import Reconstruction
from numerical_pdes.boundary_conditions import BoundaryCondition


class BurgersEquation1D(Equation):
    def __init__(self, diff_coef: float = 0.0):
        self.diff_coef = diff_coef

    @property
    def name(self) -> str:
        prefix = "Viscous" if self._is_viscous() else "Inviscid"
        return f"{prefix} Burgers Equation"

    @property
    def n_states(self) -> int:
        return 1

    def _is_viscous(self, tol: float = 1e-8) -> bool:
        if 0 <= self.diff_coef <= tol:
            return False
        elif self.diff_coef > 0:
            return True
        raise ValueError(f"diffusion coefficient ({self.diff_coef}) must be non-negative")

    def max_wave_speed(self, u: NDArray) -> float:
        return float(np.max(np.abs(u)))

    def eigenvalues(self, u: NDArray) -> Tuple[float, float]:
        return float(u[0]), float(u[0])

    def compute_cfl_dt(self, u: NDArray, dx: float, cfl_number: float) -> float:
        dt = super().compute_cfl_dt(u, dx, cfl_number)
        if self._is_viscous():
            dt = min(dt, cfl_number * dx**2 / self.diff_coef)
        return dt

    def physical_flux(self, u: NDArray) -> NDArray:
        return u**2 / 2

    def compute_numerical_flux(self, u, mesh, reconstruction, bc, n_ghost_cells):
        n_cells = mesh.n_cells
        flux = np.zeros([n_cells + 1, self.n_states])

        a_neg, a_pos, u_right, u_left = self.get_local_speed(
            u, mesh, reconstruction, bc, n_ghost_cells
        )
        f_left = self.physical_flux(u_left)
        f_right = self.physical_flux(u_right)

        for i in range(n_cells + 1):
            if (a_pos[i] - a_neg[i]) > 1e-8:
                flux[i, 0] = (
                    a_pos[i] * f_left[i, 0]
                    - a_neg[i] * f_right[i, 0]
                    + a_pos[i] * a_neg[i] * (u_right[i, 0] - u_left[i, 0])
                ) / (a_pos[i] - a_neg[i])
            else:
                flux[i, 0] = (f_left[i, 0] + f_right[i, 0]) / 2

        return flux
