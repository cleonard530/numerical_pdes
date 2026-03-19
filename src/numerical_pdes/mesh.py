import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray


class Mesh(ABC):
    @property
    @abstractmethod
    def ndim(self) -> int:
        ...

    @property
    @abstractmethod
    def n_cells(self) -> int:
        ...


class Mesh1D(Mesh):
    def __init__(self, x_left: float, x_right: float, n_cells: int):
        self._n_cells = n_cells
        self.x_left = x_left
        self.x_right = x_right
        self.dx: float = (x_right - x_left) / n_cells
        self.cell_centers: NDArray = np.linspace(
            x_left + self.dx / 2,
            x_right - self.dx / 2,
            n_cells,
        )

    @property
    def ndim(self) -> int:
        return 1

    @property
    def n_cells(self) -> int:
        return self._n_cells

    @property
    def period(self) -> float:
        return self.x_right - self.x_left
