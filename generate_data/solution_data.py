import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import List


class SolutionData:
    def __init__(self, method, n_states, labels=None, solution=None):
        self.method = method
        self.n_states = n_states
        self.labels = labels
        self.solution = solution

    def set_solution(self, solution):
        assert self.n_states == solution.shape[1], f'solution shape is not equal to {self.n_states}.'
        self.solution = solution
        if self.labels is None:
            self.labels = [f'solution {n}' for n in range(self.n_states)]

    def plot_solution(self, time_index: int, x: NDArray, marker_types: List[str] | None = None) -> None:
        if marker_types is None:
            marker_types = ['*', '-.', 'o', '.', '-']
        assert len(marker_types) >= self.n_states

        for n in range(self.n_states):
            marker_type = marker_types[n]
            label = self.labels[n]
            y = self.solution[:, n, time_index]
            plt.plot(x, y, marker_type, label=f"{self.method}-{label}")
