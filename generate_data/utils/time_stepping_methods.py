import numpy as np
from numpy.typing import NDArray
from typing import Callable

from base_solvers.base_pde_solver_1d import BasePDESolver1D


def get_next_step_o1(solver: BasePDESolver1D, u: NDArray, time_step: float, flux_function: Callable):
    t = 0
    dt = solver.set_dt()
    n_gc = solver.n_ghost_cells

    while t < time_step - 1e-10:
        dt = min(dt, time_step - t)

        flux = flux_function(u)  # self.get_uw_flux_o1(u)
        u[n_gc:-n_gc, :] = u[n_gc:-n_gc, :] - (dt / solver.dx) * (flux[1:, :] - flux[:-1, :])
        solver.set_ghost_cells(u)

        t += dt

    return u


def get_ssp_rk3_next_step(solver: BasePDESolver1D, u: NDArray, time_step: float, flux_function: Callable):
    t = 0
    dt = solver.set_dt()
    n_gc = solver.n_ghost_cells

    u1 = np.zeros_like(u)
    u2 = np.zeros_like(u)
    while t < time_step - 1e-10:
        dt = min(dt, time_step - t)

        # first stage
        flux = flux_function(u)
        u1[n_gc:-n_gc, :] = u[n_gc:-n_gc, :] - (dt / solver.dx) * (flux[1:, :] - flux[:-1, :])
        solver.set_ghost_cells(u1)

        # second stage
        flux = flux_function(u1)
        u2[n_gc:-n_gc, :] = .75*u[n_gc:-n_gc, :] + .25*(
                u1[n_gc:-n_gc, :] - (dt / solver.dx) * (flux[1:, :] - flux[:-1, :]))
        solver.set_ghost_cells(u2)

        # third stage
        flux = flux_function(u2)
        u[n_gc:-n_gc, :] = (1/3.0) * u[n_gc:-n_gc, :] + (2/3.0) * (
                u2[n_gc:-n_gc, :] - (dt / solver.dx) * (flux[1:, :] - flux[:-1, :]))
        solver.set_ghost_cells(u)

        t += dt

    return u
