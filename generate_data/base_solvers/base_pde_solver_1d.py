import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Callable, Tuple


class BasePDESolver1D(ABC):
    def __init__(self, equation: str, ax: float, bx: float, nx: int, n_states: int, bc: str = 'periodic'):
        self.equation = equation
        self.ax: float = ax
        self.bx: float = bx
        self.n_cells: int = nx
        self.dx: float = (self.bx-self.ax) / self.n_cells
        self.bc: str = bc
        self.x: NDArray = np.linspace(self.ax + (self.dx / 2), self.bx - (self.dx / 2), self.n_cells)
        self.n_ghost_cells: int | None = None
        self.n_states: int = n_states

    @abstractmethod
    def get_solution(self, u: NDArray, tspan: NDArray, method: str = 'cu2') -> NDArray:
        pass

    @abstractmethod
    def set_dt(self) -> float:
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
        for i in range(self.n_ghost_cells):
            u[i, :] = u[2 * self.n_ghost_cells - 1 - i, :]
            u[-1, :] = u[-(2 * self.n_ghost_cells) + i, :]

    def get_linear_cell_boundary_approximation(self, u: NDArray, theta: float) -> Tuple[NDArray, NDArray]:
        u_right = np.zeros([self.n_cells + 1])  # right of boundary limit
        u_left = np.zeros([self.n_cells + 1])  # left of boundary limit

        for i in range(self.n_cells):
            du = self.du_dx(u[i], u[i + 1], u[i + 2], theta)
            u_left[i + 1] = u[i + 1] + du * (self.dx / 2)
            u_right[i] = u[i + 1] - du * (self.dx / 2)
        u_left[0] = u_left[self.n_cells]
        u_right[self.n_cells] = u_right[0]

        return u_right, u_left

    def get_local_speed(self, u: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        a_negative = np.zeros((self.n_cells+1, self.n_states))
        a_positive = np.zeros((self.n_cells+1, self.n_states))

        u_right = np.zeros((self.n_cells + 1, self.n_states))
        u_left = np.zeros((self.n_cells + 1, self.n_states))
        theta = 1.2
        for n in range(self.n_states):
            u_right[:, n], u_left[:, n]  = self.get_linear_cell_boundary_approximation(u[:, n], theta)

        for i in range(0, self.n_cells+1):
            eigen_max_left, eigen_min_left = self.get_eigen_values_df_du(u_left[i, :])
            eigen_max_right, eigen_min_right = self.get_eigen_values_df_du(u_right[i, :])

            a_positive[i] = max([eigen_max_left, eigen_max_right, 0])
            a_negative[i] = min([eigen_min_left, eigen_min_right, 0])

        return a_negative, a_positive, u_right, u_left

    @staticmethod
    def minmod(forward_approx: float, backward_approx: float, central_approx: float) -> float:
        if forward_approx > 0 and backward_approx > 0:
            return min(forward_approx, backward_approx, central_approx)
        elif forward_approx < 0 and backward_approx < 0:
            return max(forward_approx, backward_approx, central_approx)
        else:
            return 0

    def du_dx(self, um1: float, u: float, up1: float, theta: float) -> float:
        forward_approx = theta*(up1-u)
        backward_approx = theta*(u-um1)
        central_approx = (up1 - um1) / 2

        derivative_approx = self.minmod(forward_approx, backward_approx, central_approx) / self.dx
        return derivative_approx

    def get_eigen_values_df_du(self, u: float) -> Tuple[float, float]:
        # implement this method for any method that uses the central upwind method
        return 0, 0

    # def plot_solution(self, solution_data: List[SolutionData], n_times_stamps: int = 4, t_max: float = None) -> None:
    #     n_solutions = len(solution_data)
    #     max_t_index = solution_data[0].solution.shape[-1] - 1
    #     t_indices = max_t_index / (n_times_stamps-1) * np.arange(0, n_times_stamps)
    #     t_indices = t_indices.astype(int)
    #     y_max = np.max(solution_data[0].solution)
    #     y_min = np.min(solution_data[0].solution)
    #     for i in range(1, n_solutions):
    #         y_max = max(y_max, np.max(solution_data[i].solution))
    #         y_min = min(y_min, np.min(solution_data[i].solution))
    #         assert solution_data[i].solution.shape == solution_data[0].solution.shape, \
    #             'solution values are of different shapes'
    #     y_range = y_max - y_min
    #
    #     ticksize = 12
    #     labelsize = 14
    #     for i in range(n_times_stamps):
    #         ti = t_indices[i]
    #         for sol in solution_data:
    #             sol.plot_solution(ti, self.x)
    #             # for j in range(self.n_states):
    #             #     sol.plot_solution(j, ti, self.x)
    #
    #         if t_max is not None:
    #             t = t_max * (ti / t_indices[-1])
    #             plt.title(f"{self.equation}: t = {t: .2f}")
    #         plt.ylim(y_min - .1*y_range, y_max + .1*y_range)
    #         plt.xlabel('x', fontsize=labelsize)
    #         plt.ylabel('y', fontsize=labelsize)
    #         plt.xticks(fontsize=ticksize)
    #         plt.yticks(fontsize=ticksize)
    #         plt.legend()
    #         plt.show()
    #
    # def plot_animation(self, solution_data: List[SolutionData], t_max: float) -> None:
    #     n_frames = solution_data[0].solution.shape[2]
    #     last_t_index = n_frames - 1
    #     y_max = np.max(solution_data[0].solution)
    #     y_min = np.min(solution_data[0].solution)
    #     for i in range(1, self.n_states):
    #         y_max = max(y_max, np.max(solution_data[i].solution))
    #         y_min = min(y_min, np.min(solution_data[i].solution))
    #         assert solution_data[i].solution.shape == solution_data[0].solution.shape, 'solution values are of different shapes'
    #     y_range = y_max - y_min
    #
    #     fig, ax = plt.subplots()
    #     ticksize = 12
    #     labelsize = 14
    #
    #     def update(frame):
    #         fig.clear()
    #         for sol in solution_data:
    #             sol.plot_solution(frame, self.x)
    #
    #         t = t_max * (frame / last_t_index)
    #         plt.title(f"{self.equation}: t = {t: .2f}")
    #
    #         plt.ylim(y_min - .1 * y_range, y_max + .1 * y_range)
    #         plt.xlabel('x', fontsize=labelsize)
    #         plt.ylabel('y', fontsize=labelsize)
    #         plt.xticks(fontsize=ticksize)
    #         plt.yticks(fontsize=ticksize)
    #         plt.legend(loc='upper right')
    #         return None
    #
    #     ani = animation.FuncAnimation(fig=fig,
    #                                   func=update,
    #                                   frames=n_frames,
    #                                   interval=100,
    #                                   repeat=False)
    #     plt.show()
    #
    # def run_with_animator(self,
    #                       u0: NDArray,
    #                       solution_data: List[SolutionData],
    #                       t_span: NDArray) -> None:
    #     self.n_states = u0.shape[-1]
    #
    #     n_frames = len(t_span)
    #     t_max = t_span[-1]
    #
    #     u = np.zeros((self.n_cells + 2, self.n_states, n_frames))
    #
    #     tspan = np.linspace(0, t_max, n_frames)
    #
    #     u = np.zeros((self.n_cells+2, 2, n_frames))
    #
    #     y_min = np.min(u0)
    #     y_max = np.max(u0)
    #     y_range = y_max-y_min
    #
    #     fig, ax = plt.subplots()
    #     ticksize = 12
    #     labelsize = 14
    #
    #     self.n_ghost_cells = 1
    #
    #     for sol in solution_data:
    #         u[1:self.n_cells + 1, :, 0] = u0
    #         sol.set_solution(u[1:self.n_cells + 1, :, :].copy())
    #         sol.plot_solution(0, self.x)
    #
    #     t = 0
    #     plt.title(f"{self.equation}: t = {t: .2f}")
    #
    #     plt.ylim(y_min - .1 * y_range, y_max + .1 * y_range)
    #     plt.xlabel('x', fontsize=labelsize)
    #     plt.ylabel('y', fontsize=labelsize)
    #     plt.xticks(fontsize=ticksize)
    #     plt.yticks(fontsize=ticksize)
    #     plt.legend(loc='upper right')
    #     plt.show(block=False)
    #
    #     u = np.zeros([self.n_cells + 2, self.n_states, n_frames])
    #
    #     def update(frame):
    #         fig.clear()
    #         if frame == n_frames-1:
    #             plt.close()
    #
    #         for sol in solution_data:
    #             u[self.n_ghost_cells:-self.n_ghost_cells, :, frame] = sol.solution[:, :, frame]
    #             self.set_ghost_cells(u[:, :, 0])
    #             time_step = tspan[frame + 1] - tspan[frame]
    #             get_next_step_method = self.get_next_step_method(sol.method)
    #             flux_function = self.get_flux_method(sol.method)
    #             u[:, :, frame + 1] = get_next_step_method(self, u[:, :, frame], time_step, flux_function)
    #             sol.solution[:, :, frame + 1] = u[1:self.n_cells + 1, :, frame+1].copy()
    #             sol.plot_solution(frame+1, self.x)
    #
    #         t = t_max * (frame+1) / (n_frames-1)
    #         plt.title(f"{self.equation}: t = {t: .2f}")
    #
    #         plt.ylim(y_min - .1 * y_range, y_max + .1 * y_range)
    #         plt.xlabel('x', fontsize=labelsize)
    #         plt.ylabel('y', fontsize=labelsize)
    #         plt.xticks(fontsize=ticksize)
    #         plt.yticks(fontsize=ticksize)
    #         plt.legend(loc='upper right')
    #         return None
    #
    #     ani = animation.FuncAnimation(fig=fig,
    #                                   func=update,
    #                                   frames=n_frames-1,
    #                                   interval=100,
    #                                   repeat=False)
    #
    #     plt.show()
    #     # ani.save("temp.mp4", fps=30, writer='ffmpeg')
    #     print('HI ')
