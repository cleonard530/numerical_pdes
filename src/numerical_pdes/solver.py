from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from typing import List

from numerical_pdes.mesh import Mesh1D
from numerical_pdes.boundary_conditions import BoundaryCondition
from numerical_pdes.reconstruction import Reconstruction
from numerical_pdes.time_integrators import TimeIntegrator
from numerical_pdes.equations.base import Equation
from numerical_pdes.solution import SolutionData


@dataclass
class PDEProblem:
    """The equation, mesh, and numerical scheme that define a simulation."""

    equation: Equation
    mesh: Mesh1D
    bc: BoundaryCondition
    time_integrator: TimeIntegrator
    reconstruction: Reconstruction
    cfl_number: float = 0.5

    @property
    def n_ghost_cells(self) -> int:
        return self.reconstruction.required_ghost_cells


def solve(
    problem: PDEProblem,
    u0: NDArray,
    tspan: NDArray,
    label: str | None = None,
    state_labels: List[str] | None = None,
) -> SolutionData:
    """Run the simulation described by `problem` and return a SolutionData object.

    Parameters
    ----------
    problem : The equation, mesh, boundary condition, time integrator, and
        reconstruction that define the simulation.
    u0 : Initial condition, shape (n_cells, n_states).
    tspan : Array of output times including t=0.
    label : Label for this run (used in plot legends).
    state_labels : Labels for each state component.
    """
    if label is None:
        label = type(problem.time_integrator).__name__

    gc = problem.n_ghost_cells
    n_cells = problem.mesh.n_cells
    n_states = problem.equation.n_states
    n_steps = len(tspan) - 1

    solution = np.zeros([n_cells, n_states, n_steps + 1])

    u = np.zeros([n_cells + 2 * gc, n_states])
    u[gc:-gc, :] = u0.copy()
    problem.bc.apply(u, gc)
    solution[:, :, 0] = u[gc:-gc, :]

    for i in range(n_steps):
        time_step = tspan[i + 1] - tspan[i]
        u = _advance(problem, u.copy(), time_step)
        solution[:, :, i + 1] = u[gc:-gc, :]
        print(f"  t = {tspan[i + 1]:.4f}")

    result = SolutionData(
        method=label,
        n_states=n_states,
        labels=state_labels,
    )
    result.set_solution(solution)
    return result


def _advance(problem: PDEProblem, u: NDArray, time_step: float) -> NDArray:
    """Advance the solution from t to t + time_step with CFL sub-stepping."""
    t = 0.0
    while t < time_step - 1e-10:
        dt = problem.equation.compute_cfl_dt(u, problem.mesh.dx, problem.cfl_number)
        dt = min(dt, time_step - t)
        u = problem.time_integrator.step(
            u, dt, lambda u: _compute_rhs(problem, u), problem.bc, problem.n_ghost_cells
        )
        t += dt
    return u


def _compute_rhs(problem: PDEProblem, u: NDArray) -> NDArray:
    """Compute the spatial right-hand side (flux divergence) on interior cells."""
    flux = problem.equation.compute_numerical_flux(
        u, problem.mesh, problem.reconstruction, problem.bc, problem.n_ghost_cells
    )
    return -(1.0 / problem.mesh.dx) * (flux[1:, :] - flux[:-1, :])
