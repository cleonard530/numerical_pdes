import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple, List

from equation_solvers import BasePDESolver1D
from solution_data import SolutionData
from utils import get_next_step_o1_explicit, get_ssp_rk3_next_step
from utils.functions import trig_function


class BurgersEquationSolver(BasePDESolver1D):
    def __init__(self, ax: float, bx: float, n_cells: int, diff_coef: float, bc: str = 'periodic'):
        super().__init__(equation='Burgers Equation', ax=ax, bx=bx, n_cells=n_cells, n_states=1, bc=bc)
        self.diff_coef = diff_coef
        if self._is_viscous():
            self.equation = f"Viscous {self.equation}"
        else:
            self.equation = f"Inviscid {self.equation}"

    def _is_viscous(self, tol=1e-8):
        if 0 <= self.diff_coef <= tol:
            return False
        elif self.diff_coef > 0:
            return True
        else:
            raise ValueError(f"diffusion coefficient ({self.diff_coef}) must be non-negative")

    def get_solution(self, u0: NDArray, tspan: NDArray, method: str = 'cu2') -> NDArray:
        if method == 'cu1':
            self.n_ghost_cells = 1
        elif method == 'cu2':
            self.n_ghost_cells = 1
        else:
            raise TypeError(f'method {method} is unrecognized.')
        u = self.run_solver(u0, tspan, self.get_next_step_method(method), self.get_flux_method(method))
        return u

    def set_dt(self, u: NDArray, is_explicit=True) -> float:
        cfl_number = 0.5
        lambda_max = np.max(np.abs(u))
        dt = cfl_number * self.dx*lambda_max
        if self._is_viscous():
            dt = min(dt, cfl_number * self.dx**2 / self.diff_coef)
        return dt

    def get_flux(self, u: NDArray):
        return u**2 / 2

    def get_cu_flux_o1(self, u: NDArray) -> NDArray:
        flux = np.zeros([self.n_cells + 1, self.n_states])  # flux at cell interface
        a_negative, a_positive, u_right, u_left = self.get_local_speed(
            u,
            self.get_constant_cell_boundary_approximation,
            theta=1)
        Fu_negative = self.get_flux(u_left)
        Fu_positive = self.get_flux(u_right)

        for i in range(self.n_cells+1):
            if (a_positive[i] - a_negative[i]) > 1e-8:
                flux[i, 0] = a_positive[i]*Fu_negative[i, 0] - a_negative[i]*Fu_positive[i, 0]
                flux[i, 0] += a_positive[i]*a_negative[i]*(u_right[i, 0] - u_left[i, 0])
                flux[i, 0] /= (a_positive[i] - a_negative[i])
            else:
                flux[i, 0] = (Fu_negative[i, 0] + Fu_positive[i, 0]) / 2

        return flux

    def get_cu_flux_o2(self, u: NDArray) -> NDArray:
        flux = np.zeros([self.n_cells + 1, self.n_states])  # flux at cell interface
        a_negative, a_positive, u_right, u_left = self.get_local_speed(
            u,
            self.get_linear_cell_boundary_approximation,
            theta=1)
        Fu_negative = self.get_flux(u_left)
        Fu_positive = self.get_flux(u_right)

        for i in range(self.n_cells + 1):
            if (a_positive[i] - a_negative[i]) > 1e-8:
                flux[i, 0] = a_positive[i] * Fu_negative[i, 0] - a_negative[i] * Fu_positive[i, 0]
                flux[i, 0] += a_positive[i] * a_negative[i] * (u_right[i, 0] - u_left[i, 0])
                flux[i, 0] /= (a_positive[i] - a_negative[i])
            else:
                flux[i, 0] = (Fu_negative[i, 0] + Fu_positive[i, 0]) / 2

        return flux

    def get_eigen_values_df_du(self, u: NDArray) -> Tuple[float, float]:
        return u[0], u[0]

    def get_next_step_method(self, key: str) -> Callable:
        methods = {'cu1': get_next_step_o1_explicit, 'cu2': get_ssp_rk3_next_step}
        return methods[key]

    def get_flux_method(self, key: str) -> Callable:
        flux_methods = {'cu1': self.get_cu_flux_o1, 'cu2': self.get_cu_flux_o2}
        return flux_methods[key]

    def get_true_solution(self):
        pass
