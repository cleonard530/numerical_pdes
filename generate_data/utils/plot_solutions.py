import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List
import numpy as np
from numpy.typing import NDArray

from solution_data import SolutionData
from equation_solvers import BasePDESolver1D


class SolutionPlotter:
    def __init__(self, ax: float, bx: float, fig_num: int = 1, ticksize: int = 12, labelsize: int = 12):
        self.ax: float = ax
        self.bx: float = bx
        self.fig_num = fig_num
        self.fig, self.axes = None, None
        self.ticksize = ticksize
        self.labelsize = labelsize

    def plot_solutions(self, solution_data: List[SolutionData],
                       title: str,
                       t_max: float,
                       n_times_stamps: int = 4) -> None:
        n_cells = solution_data[0].solution.shape[0]
        x = self._get_x(n_cells)
        max_t_index = solution_data[0].solution.shape[-1] - 1
        t_indices = max_t_index / (n_times_stamps-1) * np.arange(0, n_times_stamps)
        t_indices = t_indices.astype(int)
        y_min, y_max = self._get_y_min_and_max(solution_data)

        for i in range(n_times_stamps):
            ti = t_indices[i]
            for sol in solution_data:
                sol.plot_solution(ti, x)

            t = t_max * (ti / t_indices[-1])
            t_title = f"{title}: t = {t: .2f}"
            self._set_plot_metadata(title=t_title,
                                    y_min=y_min,
                                    y_max=y_max)
            plt.show()

    def plot_animation(self, solution_data: List[SolutionData], title: str, t_max: float) -> None:
        n_frames = solution_data[0].solution.shape[2]
        n_cells = solution_data[0].solution.shape[0]
        x = self._get_x(n_cells)

        last_t_index = n_frames - 1
        y_min, y_max = self._get_y_min_and_max(solution_data)

        fig, ax = plt.subplots()

        def update(frame):
            fig.clear()
            for sol in solution_data:
                sol.plot_solution(frame, x)

            t = t_max * (frame / last_t_index)
            self._set_plot_metadata(title=f"{title}: t = {t: .2f}",
                                    y_min=y_min,
                                    y_max=y_max)
            return None

        ani = animation.FuncAnimation(fig=fig,
                                      func=update,
                                      frames=n_frames,
                                      interval=100,
                                      repeat=False)
        plt.show()

    def run_with_animator(self,
                          solver: BasePDESolver1D,
                          u0: NDArray,
                          solution_data: List[SolutionData],
                          t_span: NDArray) -> None:
        n_states = solution_data[0].n_states

        n_frames = len(t_span)
        t_max = t_span[-1]

        u = np.zeros((solver.n_cells + 2, n_states, n_frames))

        tspan = np.linspace(0, t_max, n_frames)

        u = np.zeros((solver.n_cells+2, 2, n_frames))

        y_min = np.min(u0)
        y_max = np.max(u0)
        y_range = y_max - y_min

        fig, ax = plt.subplots()

        solver.n_ghost_cells = 1

        for sol in solution_data:
            u[1:solver.n_cells + 1, :, 0] = u0
            sol.set_solution(u[1:solver.n_cells + 1, :, :].copy())
            sol.plot_solution(0, solver.x)

        t = 0
        self._set_plot_metadata(title=f"{solver.equation}: t = {t: .2f}",
                                y_min=y_min - .1 * y_range,
                                y_max=y_max + .1 * y_range)
        plt.show(block=False)

        u = np.zeros([solver.n_cells + 2, n_states, n_frames])

        def update(frame):
            fig.clear()
            if frame == n_frames - 1:
                plt.close()

            for sol in solution_data:
                u[solver.n_ghost_cells:-solver.n_ghost_cells, :, frame] = sol.solution[:, :, frame]
                solver.set_ghost_cells(u[:, :, 0])
                time_step = tspan[frame + 1] - tspan[frame]
                next_step_method = solver.get_next_step_method(sol.method)
                flux_function = solver.get_flux_method(sol.method)
                u[:, :, frame + 1] = next_step_method(solver, u[:, :, frame], time_step, flux_function)
                sol.solution[:, :, frame + 1] = u[1:solver.n_cells + 1, :, frame+1].copy()
                sol.plot_solution(frame+1, solver.x)

            t = t_max * (frame+1) / (n_frames-1)
            self._set_plot_metadata(title=f"{solver.equation}: t = {t: .2f}",
                                    y_min=y_min - .1 * y_range,
                                    y_max=y_max + .1 * y_range)
            return None

        ani = animation.FuncAnimation(fig=fig,
                                      func=update,
                                      frames=n_frames-1,
                                      interval=100,
                                      repeat=False)

        plt.show()
        # ani.save("temp.mp4", fps=30, writer='ffmpeg')

    def _get_y_min_and_max(self, solution_data: List[SolutionData], range_amplifier=0.1):
        n_solutions = len(solution_data)
        y_max = np.max(solution_data[0].solution)
        y_min = np.min(solution_data[0].solution)
        for i in range(1, n_solutions):
            y_max = max(y_max, np.max(solution_data[i].solution))
            y_min = min(y_min, np.min(solution_data[i].solution))
            assert solution_data[i].solution.shape == solution_data[
                0].solution.shape, 'solution values are of different shapes'
        y_range = y_max - y_min
        y_min -= range_amplifier * y_range
        y_max += range_amplifier * y_range
        return y_min, y_max

    def _get_x(self, n_cells: int):
        dx = (self.bx - self.ax) / n_cells
        x = np.linspace(self.ax + (dx / 2), self.bx - (dx / 2), n_cells)
        return x

    def _set_plot_metadata(self, title: str, y_min: float, y_max: float, legend_loc='upper right'):
        plt.title(title)

        plt.ylim(y_min, y_max)
        plt.xlabel('x', fontsize=self.labelsize)
        plt.ylabel('y', fontsize=self.labelsize)
        plt.xticks(fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.legend(loc=legend_loc)
