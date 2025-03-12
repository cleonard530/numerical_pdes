import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Callable, Tuple


class BasePDESolver1D(ABC):
    def __init__(self, equation: str, ax: float, bx: float, n_cells: int, n_states: int, bc: str = 'periodic'):
        self.equation = equation
        self.ax: float = ax
        self.bx: float = bx
        self.n_cells: int = n_cells
        self.dx: float = (self.bx-self.ax) / self.n_cells
        self.bc: str = bc
        self.x: NDArray = np.linspace(self.ax + (self.dx / 2), self.bx - (self.dx / 2), self.n_cells)
        self.n_ghost_cells: int | None = None
        self.n_states: int = n_states

    @abstractmethod
    def get_solution(self, u: NDArray, tspan: NDArray, method: str = 'cu2') -> NDArray:
        pass

    @abstractmethod
    def set_dt(self, u: NDArray, is_explicit=True) -> float:
        pass

    @abstractmethod
    def get_next_step_method(self, key: str) -> Callable:
        pass

    @abstractmethod
    def get_flux_method(self, key: str) -> Callable:
        pass

    def run_solver(self,
                   u0: NDArray,
                   tspan: NDArray,
                   get_next_step_method: Callable,
                   flux_function: Callable) -> NDArray:
        n_steps = len(tspan) - 1
        u = np.zeros([self.n_cells + 2*self.n_ghost_cells, self.n_states, n_steps + 1])

        u[self.n_ghost_cells:-self.n_ghost_cells, :, 0] = u0.copy()
        self.set_ghost_cells(u[:, :, 0])

        for i in range(n_steps):
            time_step = tspan[i + 1] - tspan[i]
            u[:, :, i + 1] = get_next_step_method(self, u[:, :, i].copy(), time_step, flux_function)
            print(f"{tspan[i+1] =}")
        return u[self.n_ghost_cells:-self.n_ghost_cells, :, :]

    def set_ghost_cells(self, u: NDArray) -> None:
        assert self.bc in ('periodic', 'wall'), f"{self.bc} is an invalid boundary condition"
        if self.bc == 'periodic':
            self.periodic_ghost_cell(u)

        if self.bc == 'wall':
            self.wall_ghost_cell(u)

    def periodic_ghost_cell(self, u: NDArray) -> None:
        for i in range(self.n_ghost_cells):
            u[i, :] = u[i+self.n_cells, :]
            u[-1-i, :] = u[-1-i-self.n_cells, :]

    def wall_ghost_cell(self, u: NDArray) -> None:
        # will need to update this for a lot of equations (such as Euler's equation)
        for i in range(self.n_ghost_cells):
            u[i, :] = u[2 * self.n_ghost_cells - 1 - i, :]
            u[-1, :] = u[-(2 * self.n_ghost_cells) + i, :]

    def set_boundary_wall_ghost_cells(self, u_right: NDArray, u_left: NDArray) -> None:
        # will need to update this for a lot of equations (such as Euler's equation)
        assert self.bc in ('periodic', 'wall'), f"{self.bc} is an invalid boundary condition"
        if self.bc == 'periodic':
            u_left[0] = u_left[self.n_cells]
            u_right[self.n_cells] = u_right[0]

        if self.bc == 'wall':
            u_left[0] = u_right[0]
            u_right[self.n_cells] = u_left[self.n_cells]

    def get_linear_cell_boundary_approximation(self, u: NDArray, theta: float) -> Tuple[NDArray, NDArray]:
        u_right = np.zeros([self.n_cells + 1])  # right of boundary limit
        u_left = np.zeros([self.n_cells + 1])  # left of boundary limit

        for i in range(self.n_cells):
            offset = self.n_ghost_cells - 1
            du = self.du_dx(u[offset + i], u[offset + i + 1], u[offset + i + 2], theta)
            u_left[i + 1] = u[offset + i + 1] + du * (self.dx / 2)
            u_right[i] = u[offset + i + 1] - du * (self.dx / 2)
        self.set_boundary_wall_ghost_cells(u_right=u_right, u_left=u_left)

        return u_right, u_left

    def get_constant_cell_boundary_approximation(self, u: NDArray, theta: float) -> Tuple[NDArray, NDArray]:
        u_right = np.zeros([self.n_cells + 1])  # right of boundary limit
        u_left = np.zeros([self.n_cells + 1])  # left of boundary limit

        for i in range(self.n_cells+1):
            offset = self.n_ghost_cells - 1
            u_left[i] = u[offset + i + 1]
            u_right[i] = u[offset + i]
        self.set_boundary_wall_ghost_cells(u_right=u_right, u_left=u_left)

        return u_right, u_left

    def get_local_speed(self, u: NDArray,
                        cell_boundary_approximation: Callable,
                        theta: float = 1.0) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        a_negative = np.zeros((self.n_cells+1, self.n_states))
        a_positive = np.zeros((self.n_cells+1, self.n_states))

        u_right = np.zeros((self.n_cells + 1, self.n_states))
        u_left = np.zeros((self.n_cells + 1, self.n_states))
        for n in range(self.n_states):
            u_right[:, n], u_left[:, n]  = cell_boundary_approximation(u[:, n], theta)  # self.get_linear_cell_boundary_approximation(u[:, n], theta)

        for i in range(0, self.n_cells+1):
            eigen_max_left, eigen_min_left = self.get_eigen_values_df_du(u_left[i, :])
            eigen_max_right, eigen_min_right = self.get_eigen_values_df_du(u_right[i, :])

            a_positive[i] = max([eigen_max_left, eigen_max_right, 0])
            a_negative[i] = min([eigen_min_left, eigen_min_right, 0])

        return a_negative, a_positive, u_right, u_left

    def du_dx(self, um1: float, u: float, up1: float, theta: float) -> float:
        forward_approx = theta*(up1-u)
        backward_approx = theta*(u-um1)
        central_approx = (up1 - um1) / 2

        derivative_approx = self.minmod(forward_approx, backward_approx, central_approx) / self.dx
        return derivative_approx

    @staticmethod
    def minmod(forward_approx: float, backward_approx: float, central_approx: float) -> float:
        if forward_approx > 0 and backward_approx > 0:
            return min(forward_approx, backward_approx, central_approx)
        elif forward_approx < 0 and backward_approx < 0:
            return max(forward_approx, backward_approx, central_approx)
        else:
            return 0

    def get_eigen_values_df_du(self, u: NDArray) -> Tuple[float, float]:
        # implement this method for any method that uses the central upwind method
        return 0, 0
