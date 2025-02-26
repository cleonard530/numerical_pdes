import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class BasePDESolver1D(ABC):
    def __init__(self, ax, bx, nx, bc='periodic'):
        self.ax = ax
        self.bx = bx
        self.n_cells = nx
        self.dx = (self.bx-self.ax) / self.n_cells
        self.bc = bc
        self.x = np.linspace(self.ax + (self.dx / 2), self.bx - (self.dx / 2), self.n_cells)
        self.order = None

    @abstractmethod
    def get_solution(self, u, tspan, method='cu2'):
        pass

    def set_ghost_cells(self, u):
        assert self.bc in ('periodic', 'wall'), f"{self.bc} is an invalid boundary condition"
        if self.bc == 'periodic':
            self.periodic_ghost_cell(u)

        if self.bc == 'wall':
            self.wall_ghost_cell(u)

    def periodic_ghost_cell(self, u):
        for i in range(self.order):
            u[i, :] = u[i+self.n_cells, :]
            u[-1-i, :] = u[-1-i-self.n_cells, :]

    def wall_ghost_cell(self, u):
        for i in range(self.order):
            u[i, :] = u[2*self.order-1-i, :]
            u[-1, :] = u[-(2*self.order)+i, :]

    def trig_initial_condition(self, a=None, b=None, k=3):
       if a is None:
           a = np.random.uniform(-1, 1, k)
       if b is None:
           b = np.random.uniform(-1, 1, k)
       period = self.bx - self.ax

       y = np.zeros(self.x.shape)
       for i in range(1, k + 1):
          y = y + (1 / i) * (a[i - 1] * np.cos(2*np.pi*i*self.x/period) +
                             b[i - 1] * np.sin(2*np.pi*i*self.x/period))
       return y

    @staticmethod
    def minmod(f, b, c):
        if f > 0 and b > 0:
            return min(f, b, c)
        if f < 0 and b < 0:
            return max(f, b, c)

        return 0

    def dudx(self, um1, u, up1, h, theta):
        f = theta*(up1-u)/h
        b = theta*(u-um1)/h
        c = (up1 - um1) / (2 * h)

        d = self.minmod(f, b, c)
        return d

    def plot_solution(self, ys, n_times_stamps=4, t_max=None):
        n_solutions = len(ys)
        max_index = ys[0].shape[1]-1
        t_indices = max_index / (n_times_stamps-1) * np.arange(0, n_times_stamps)
        t_indices = t_indices.astype(int)
        y_max = np.max(ys[0])
        y_min = np.min(ys[0])
        for i in range(1, n_solutions):
            y_max = max(y_max, np.max(ys[i]))
            y_min = min(y_min, np.min(ys[i]))
            assert ys[i].shape == ys[0].shape, 'solution values are of different shapes'
        y_range = y_max - y_min

        ticksize = 12
        labelsize = 14
        for i in range(n_times_stamps):
            ti = t_indices[i]
            for y in ys:
                yi = y[:, ti]
                plt.plot(self.x, yi, '.', label=f'{i}')
            if t_max is not None:
                t = t_max * (ti / t_indices[-1])
                plt.title(f"t = {t: .2f}")
            plt.ylim(y_min - .1*y_range, y_max + .1*y_range)
            plt.xlabel('x', fontsize=labelsize)
            plt.ylabel('y', fontsize=labelsize)
            plt.xticks(fontsize=ticksize)
            plt.yticks(fontsize=ticksize)
            plt.legend()
            plt.show()
