import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple, List

from equation_solvers import BasePDESolver1D
from solution_data import SolutionData
from utils import get_next_step_o1_explicit, get_ssp_rk3_next_step
from utils.functions import trig_function


class WaveEquationSolver(BasePDESolver1D):
    def __init__(self, ax: float, bx: float, n_cells: int, wave_speed: float, bc: str = 'periodic'):
        super().__init__(equation='Wave Equation', ax=ax, bx=bx, n_cells=n_cells, n_states=2, bc=bc)
        self.wave_speed = wave_speed

    def get_solution(self, u0: NDArray, tspan: NDArray, method: str = 'cu2') -> NDArray:
        if method == 'uw1':
            self.n_ghost_cells = 1
            u = self.run_solver(u0, tspan, get_next_step_o1_explicit, self.get_uw_flux_o1)
        elif method == 'uw2':
            self.n_ghost_cells = 1
            u = self.run_solver(u0, tspan, get_ssp_rk3_next_step, self.get_uw_flux_o2)
        else:
            raise TypeError(f'method {method} is unrecognized.')
        return u

    def set_dt(self, u: NDArray, is_explicit=True) -> float:

        dt = .5 * self.dx / self.wave_speed
        return dt

    def get_uw_flux_o1(self, u: NDArray) -> NDArray:
        flux = np.zeros([self.n_cells + 1, 2])  # flux at cell interface

        flux[:, 0] = (1 / 2) * (self.wave_speed * u[:-1, 0] - u[:-1, 1] -
                                self.wave_speed * u[1:, 0] - u[1:, 1])
        flux[:, 1] = (self.wave_speed / 2) * (-self.wave_speed * u[:-1, 0] + u[:-1, 1] -
                                              self.wave_speed * u[1:, 0] - u[1:, 1])
        return flux

    def get_uw_flux_o2(self, u: NDArray) -> NDArray:
        flux = np.zeros([self.n_cells + 1, 2])  # flux at cell interface

        vp, vn = self.get_linear_cell_boundary_approximation(u[:, 0], 1.0)
        wp, wn = self.get_linear_cell_boundary_approximation(u[:, 1], 1.0)

        flux[:, 0] = (1 / 2) * (self.wave_speed * vn - wn - self.wave_speed * vp - wp)
        flux[:, 1] = (self.wave_speed / 2) * (-self.wave_speed * vn + wn - self.wave_speed * vp - wp)

        return flux

    def get_cu2_flux_o2(self, u: NDArray) -> NDArray:
        flux = np.zeros([self.n_cells + 1, 2])  # flux at cell interface

        vp, vn = self.get_linear_cell_boundary_approximation(u[:, 0], 1.0)
        wp, wn = self.get_linear_cell_boundary_approximation(u[:, 1], 1.0)

        flux[:, 0] = (1 / 2) * (self.wave_speed * vn - wn - self.wave_speed * vp - wp)
        flux[:, 1] = (self.wave_speed / 2) * (-self.wave_speed * vn + wn - self.wave_speed * vp - wp)

        return flux

    def get_eigen_values_df_du(self, u: NDArray) -> Tuple[float, float]:
        # implement this method for any method that uses the central upwind method
        sqrt_c = np.sqrt(self.wave_speed)
        return sqrt_c, -sqrt_c

    def get_next_step_method(self, key: str) -> Callable:
        methods = {'uw1': get_next_step_o1_explicit, 'uw2': get_ssp_rk3_next_step}
        return methods[key]

    def get_flux_method(self, key: str) -> Callable:
        flux_methods = {'uw1': self.get_uw_flux_o1, 'uw2': self.get_uw_flux_o2}
        return flux_methods[key]

    def get_true_solution(self,
                          a1: NDArray | List,
                          b1: NDArray | List,
                          a2: NDArray | List,
                          b2: NDArray | List,
                          tspan: NDArray):
        ux = np.zeros((len(self.x), len(tspan)))
        ut = np.zeros((len(self.x), len(tspan)))
        period = self.bx-self.ax

        for j, t in enumerate(tspan):
            ux[:, j] = .5 * (trig_function(self.x, a=a1, b=b1, period=period, t=-t) +
                             trig_function(self.x, a=a1, b=b1, period=period, t=t) +
                             trig_function(self.x, a=a2, b=b2, period=period, t=t) -
                             trig_function(self.x, a=a2, b=b2, period=period, t=-t))

            ut[:, j] = .5 * (trig_function(self.x, a=a1, b=b1, period=period, t=t) -
                             trig_function(self.x, a=a1, b=b1, period=period, t=-t) +
                             trig_function(self.x, a=a2, b=b2, period=period, t=t) +
                             trig_function(self.x, a=a2, b=b2, period=period, t=-t))

        u = np.stack((ux, ut), axis=1)

        return SolutionData('true', n_states=2, labels=['ux', 'ut'], solution=u)
