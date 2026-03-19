import matplotlib.pyplot as plt
from typing import List
from numpy.typing import NDArray


class SolutionData:
    def __init__(
        self,
        method: str,
        n_states: int,
        labels: List[str] | None = None,
        solution: NDArray | None = None,
    ):
        self.method = method
        self.n_states = n_states
        self.labels = labels
        self.solution = solution

    def set_solution(self, solution: NDArray) -> None:
        assert self.n_states == solution.shape[1], (
            f"solution shape is not equal to {self.n_states}."
        )
        self.solution = solution
        if self.labels is None:
            self.labels = [f"solution {n}" for n in range(self.n_states)]

    def plot_solution(
        self,
        time_index: int,
        x: NDArray,
        marker_types: List[str] | None = None,
    ) -> None:
        if marker_types is None:
            marker_types = ["*", "-.", "o", ".", "-"]
        assert len(marker_types) >= self.n_states

        for n in range(self.n_states):
            plt.plot(
                x,
                self.solution[:, n, time_index],
                marker_types[n],
                label=f"{self.method}-{self.labels[n]}",
            )
