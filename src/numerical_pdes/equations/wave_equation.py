import numpy as np
from typing import Tuple, List
from numpy.typing import NDArray

from numerical_pdes.equations.base import Equation
from numerical_pdes.mesh import Mesh1D
from numerical_pdes.reconstruction import Reconstruction
from numerical_pdes.boundary_conditions import BoundaryCondition
from numerical_pdes.utils.functions import trig_function


class WaveEquation1D(Equation):
    def __init__(self, wave_speed: float):
        self.wave_speed = wave_speed

    @property
    def name(self) -> str:
        return "Wave Equation"

    @property
    def n_states(self) -> int:
        return 2

    def max_wave_speed(self, u: NDArray) -> float:
        return self.wave_speed

    def eigenvalues(self, u: NDArray) -> Tuple[float, float]:
        c = np.sqrt(self.wave_speed)
        return c, -c

    def compute_numerical_flux(self, u, mesh, reconstruction, bc, n_ghost_cells):
        n_cells = mesh.n_cells
        flux = np.zeros([n_cells + 1, 2])

        vp, vn = reconstruction.reconstruct(u[:, 0], mesh, n_ghost_cells, bc)
        wp, wn = reconstruction.reconstruct(u[:, 1], mesh, n_ghost_cells, bc)

        c = self.wave_speed
        flux[:, 0] = (1 / 2) * (c * vn - wn - c * vp - wp)
        flux[:, 1] = (c / 2) * (-c * vn + wn - c * vp - wp)

        return flux

    def get_true_solution(
        self,
        mesh: Mesh1D,
        a1: NDArray | List,
        b1: NDArray | List,
        a2: NDArray | List,
        b2: NDArray | List,
        tspan: NDArray,
    ) -> NDArray:
        """Compute the exact d'Alembert solution. Returns array of shape (n_cells, 2, n_times)."""
        x = mesh.cell_centers
        period = mesh.period

        ux = np.zeros((len(x), len(tspan)))
        ut = np.zeros((len(x), len(tspan)))

        for j, t in enumerate(tspan):
            ux[:, j] = 0.5 * (
                trig_function(x, a=a1, b=b1, period=period, t=-t)
                + trig_function(x, a=a1, b=b1, period=period, t=t)
                + trig_function(x, a=a2, b=b2, period=period, t=t)
                - trig_function(x, a=a2, b=b2, period=period, t=-t)
            )
            ut[:, j] = 0.5 * (
                trig_function(x, a=a1, b=b1, period=period, t=t)
                - trig_function(x, a=a1, b=b1, period=period, t=-t)
                + trig_function(x, a=a2, b=b2, period=period, t=t)
                + trig_function(x, a=a2, b=b2, period=period, t=-t)
            )

        return np.stack((ux, ut), axis=1)
