import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List
import numpy as np
from numpy.typing import NDArray
import os
import re

from numerical_pdes.mesh import Mesh1D
from numerical_pdes.solution import SolutionData


class SolutionPlotter:
    def __init__(
        self,
        mesh: Mesh1D,
        ticksize: int = 12,
        labelsize: int = 12,
        output_dir: str = "plots",
    ):
        self.mesh = mesh
        self.ticksize = ticksize
        self.labelsize = labelsize
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        name = name.lower().strip()
        name = re.sub(r"[^\w\s-]", "", name)
        name = re.sub(r"[\s]+", "_", name)
        return name

    def plot_solutions(
        self,
        solution_data: List[SolutionData],
        title: str,
        t_max: float,
        n_time_stamps: int = 4,
    ) -> None:
        x = self.mesh.cell_centers
        max_t_index = solution_data[0].solution.shape[-1] - 1
        t_indices = max_t_index / (n_time_stamps - 1) * np.arange(0, n_time_stamps)
        t_indices = t_indices.astype(int)
        y_min, y_max = self._get_y_min_and_max(solution_data)

        for i in range(n_time_stamps):
            ti = t_indices[i]
            for sol in solution_data:
                sol.plot_solution(ti, x)

            t = t_max * (ti / t_indices[-1])
            t_title = f"{title}: t = {t: .2f}"
            self._set_plot_metadata(title=t_title, y_min=y_min, y_max=y_max)
            filename = f"{self._sanitize_filename(title)}_t{t:.2f}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved: {filepath}")

    def plot_animation(
        self,
        solution_data: List[SolutionData],
        title: str,
        t_max: float,
    ) -> None:
        x = self.mesh.cell_centers
        n_frames = solution_data[0].solution.shape[2]
        last_t_index = n_frames - 1
        y_min, y_max = self._get_y_min_and_max(solution_data)

        fig, _ = plt.subplots()

        def update(frame):
            fig.clear()
            for sol in solution_data:
                sol.plot_solution(frame, x)
            t = t_max * (frame / last_t_index)
            self._set_plot_metadata(
                title=f"{title}: t = {t: .2f}", y_min=y_min, y_max=y_max
            )

        ani = animation.FuncAnimation(
            fig=fig, func=update, frames=n_frames, interval=100, repeat=False
        )
        filename = f"{self._sanitize_filename(title)}_animation.gif"
        filepath = os.path.join(self.output_dir, filename)
        ani.save(filepath, writer="pillow", fps=10)
        plt.close()
        print(f"Saved: {filepath}")

    def _get_y_min_and_max(
        self, solution_data: List[SolutionData], range_amplifier: float = 0.1
    ):
        y_max = np.max(solution_data[0].solution)
        y_min = np.min(solution_data[0].solution)
        for i in range(1, len(solution_data)):
            y_max = max(y_max, np.max(solution_data[i].solution))
            y_min = min(y_min, np.min(solution_data[i].solution))
            assert solution_data[i].solution.shape == solution_data[0].solution.shape, (
                "solution values are of different shapes"
            )
        y_range = y_max - y_min
        y_min -= range_amplifier * y_range
        y_max += range_amplifier * y_range
        return y_min, y_max

    def _set_plot_metadata(
        self, title: str, y_min: float, y_max: float, legend_loc: str = "upper right"
    ):
        plt.title(title)
        plt.ylim(y_min, y_max)
        plt.xlabel("x", fontsize=self.labelsize)
        plt.ylabel("y", fontsize=self.labelsize)
        plt.xticks(fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.legend(loc=legend_loc)
