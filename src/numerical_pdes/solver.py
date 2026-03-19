import numpy as np
from numpy.typing import NDArray
from typing import List

from numerical_pdes.mesh import Mesh1D
from numerical_pdes.boundary_conditions import BoundaryCondition
from numerical_pdes.reconstruction import Reconstruction
from numerical_pdes.time_integrators import TimeIntegrator
from numerical_pdes.equations.base import Equation
from numerical_pdes.solution import SolutionData


class PDESolver:
    def __init__(
        self,
        equation: Equation,
        mesh: Mesh1D,
        bc: BoundaryCondition,
        time_integrator: TimeIntegrator,
        reconstruction: Reconstruction,
        cfl_number: float = 0.5,
    ):
        self.equation = equation
        self.mesh = mesh
        self.bc = bc
        self.time_integrator = time_integrator
        self.reconstruction = reconstruction
        self.n_ghost_cells = reconstruction.required_ghost_cells
        self.cfl_number = cfl_number

    def solve(
        self,
        u0: NDArray,
        tspan: NDArray,
        label: str | None = None,
        state_labels: List[str] | None = None,
    ) -> SolutionData:
        """Run the simulation and return a SolutionData object.

        Parameters
        ----------
        u0 : Initial condition, shape (n_cells, n_states).
        tspan : Array of output times including t=0.
        label : Label for this run (used in plot legends).
        state_labels : Labels for each state component.
        """
        if label is None:
            label = type(self.time_integrator).__name__

        gc = self.n_ghost_cells
        n_cells = self.mesh.n_cells
        n_states = self.equation.n_states
        n_steps = len(tspan) - 1

        solution = np.zeros([n_cells, n_states, n_steps + 1])

        u = np.zeros([n_cells + 2 * gc, n_states])
        u[gc:-gc, :] = u0.copy()
        self.bc.apply(u, gc)
        solution[:, :, 0] = u[gc:-gc, :]

        for i in range(n_steps):
            time_step = tspan[i + 1] - tspan[i]
            u = self._advance(u.copy(), time_step)
            solution[:, :, i + 1] = u[gc:-gc, :]
            print(f"  t = {tspan[i + 1]:.4f}")

        result = SolutionData(
            method=label,
            n_states=n_states,
            labels=state_labels,
        )
        result.set_solution(solution)
        return result

    def _advance(self, u: NDArray, time_step: float) -> NDArray:
        """Advance the solution from t to t + time_step with CFL sub-stepping."""
        t = 0.0
        while t < time_step - 1e-10:
            dt = self.equation.compute_cfl_dt(u, self.mesh.dx, self.cfl_number)
            dt = min(dt, time_step - t)
            u = self.time_integrator.step(
                u, dt, self._compute_rhs, self.bc, self.n_ghost_cells
            )
            t += dt
        return u

    def _compute_rhs(self, u: NDArray) -> NDArray:
        """Compute the spatial right-hand side (flux divergence) on interior cells."""
        flux = self.equation.compute_numerical_flux(
            u, self.mesh, self.reconstruction, self.bc, self.n_ghost_cells
        )
        return -(1.0 / self.mesh.dx) * (flux[1:, :] - flux[:-1, :])
