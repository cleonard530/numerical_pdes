# Numerical PDEs

A Python library for numerically solving PDEs, built from composable pieces: a mesh, a boundary condition, a spatial reconstruction, a time integrator, and an equation.

Two equations are currently implemented (both 1D, finite-volume):

- **Wave equation** (`WaveEquation1D`) — solved with an upwind numerical flux; has a known exact solution for checking accuracy.
- **Burgers' equation** (`BurgersEquation1D`, inviscid) — solved with a Kurganov–Tadmor (central-upwind) numerical flux.

Each can be run at first order (`ConstantReconstruction` + `ForwardEuler`) or second order (`MinmodLinearReconstruction` + `SSPRK3`).

## Repository layout

```text
.
├── src/numerical_pdes/        # The package
│   ├── mesh.py                 # Mesh1D: uniform 1D cell-centered grid
│   ├── boundary_conditions.py  # PeriodicBC, WallBC
│   ├── reconstruction.py       # ConstantReconstruction, MinmodLinearReconstruction
│   ├── time_integrators.py     # ForwardEuler, SSPRK3
│   ├── equations/               # WaveEquation1D, BurgersEquation1D
│   ├── solver.py                # PDEProblem + solve(): runs the time-stepping loop
│   ├── solution.py              # SolutionData: stores a simulation's output
│   ├── plotting.py              # SolutionPlotter: snapshot PNGs and animated GIFs
│   └── utils/                   # Initial-condition generators and L1/L2 error metrics
├── scripts/run_simulation.py  # Runnable demo: wave equation and Burgers' equation
├── generate_data/              # Older, legacy implementation of the same solvers (kept for reference)
├── pyproject.toml              # Package metadata (name: numerical-pdes)
└── requirements.txt            # Pinned deps for the devcontainer
```

## Installation

Requires Python 3.10+.

```bash
git clone <this-repo>
cd time-stepping-neural-network
pip install -e .
```

## Running the simulation

```bash
python scripts/run_simulation.py
```

This solves the wave equation and Burgers' equation at first and second order, prints L2 errors against the exact solution (wave equation only), and saves plots to `scripts/plots/`.
