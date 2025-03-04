import matplotlib.pyplot as plt
from numpy.typing import NDArray

class SolutionData:
    def __init__(self, method, labels=None):
        self.method = method
        self.solution = None
        self.labels = labels

    def set_solution(self, solution):
        self.solution = solution
        if self.labels is None:
            n_states = solution.shape[1]
            self.labels = [f'solution {i}' for i in range(n_states)]

    def plot_solution(self, state_index: int, time_index: int, x: NDArray):
        label = self.labels[state_index]
        y = self.solution[:, state_index, time_index]
        plt.plot(x, y, '.', label=f"{self.method}-{label}")
