import numpy as np

from numerical_pdes import (
    Mesh1D,
    PeriodicBC,
    WallBC,
    ConstantReconstruction,
    MinmodLinearReconstruction,
    ForwardEuler,
    SSPRK3,
    WaveEquation1D,
    BurgersEquation1D,
    PDEProblem,
    solve,
    SolutionData,
    SolutionPlotter,
)
from numerical_pdes.utils import trig_function, piecewise_constant, l2_error


def main_wave_equation():
    mesh = Mesh1D(x_left=-np.pi, x_right=np.pi, n_cells=200)
    bc = PeriodicBC()
    equation = WaveEquation1D(wave_speed=1.0)

    a1 = [-0.4, 1.0, -0.4]
    b1 = [-0.6, -0.8, 0.7]
    ux0 = trig_function(mesh.cell_centers, a=a1, b=b1, period=mesh.period)

    a2 = [0.1, 0.1, 0.7]
    b2 = [0.4, -0.7, -0.4]
    ut0 = trig_function(mesh.cell_centers, a=a2, b=b2, period=mesh.period)

    u0 = np.stack((ux0, ut0), axis=1)
    t_max = 5
    n_steps = 20
    tspan = np.linspace(0, t_max, n_steps + 1)

    problem_o1 = PDEProblem(
        equation=equation,
        mesh=mesh,
        bc=bc,
        time_integrator=ForwardEuler(),
        reconstruction=ConstantReconstruction(),
    )
    problem_o2 = PDEProblem(
        equation=equation,
        mesh=mesh,
        bc=bc,
        time_integrator=SSPRK3(),
        reconstruction=MinmodLinearReconstruction(),
    )

    print("Running wave equation (O1)...")
    result_o1 = solve(problem_o1, u0, tspan, label="uw1", state_labels=["ux", "ut"])
    print("Running wave equation (O2)...")
    result_o2 = solve(problem_o2, u0, tspan, label="uw2", state_labels=["ux", "ut"])

    true_sol_array = equation.get_true_solution(mesh, a1, b1, a2, b2, tspan)
    result_true = SolutionData("true", n_states=2, labels=["ux", "ut"], solution=true_sol_array)

    print(f"ux O1 L2 error: {l2_error(result_o1.solution[:, 0, -1], result_true.solution[:, 0, -1]):.6e}")
    print(f"ut O1 L2 error: {l2_error(result_o1.solution[:, 1, -1], result_true.solution[:, 1, -1]):.6e}")
    print(f"ux O2 L2 error: {l2_error(result_o2.solution[:, 0, -1], result_true.solution[:, 0, -1]):.6e}")
    print(f"ut O2 L2 error: {l2_error(result_o2.solution[:, 1, -1], result_true.solution[:, 1, -1]):.6e}")

    plotter = SolutionPlotter(mesh)
    plotter.plot_solutions(
        [result_o1, result_o2, result_true],
        title=equation.name,
        n_time_stamps=4,
        t_max=t_max,
    )


def main_burgers_equation():
    nx = 500
    mesh = Mesh1D(x_left=-np.pi, x_right=np.pi, n_cells=nx)
    bc = WallBC()
    equation = BurgersEquation1D(diff_coef=0)

    u0 = np.zeros((nx, 1))
    u0[:, 0] = piecewise_constant(nx, u_left=-0.2, u_right=0.6)

    t_max = 1
    n_steps = 10
    tspan = np.linspace(0, t_max, n_steps + 1)

    problem_o1 = PDEProblem(
        equation=equation,
        mesh=mesh,
        bc=bc,
        time_integrator=ForwardEuler(),
        reconstruction=ConstantReconstruction(),
    )
    problem_o2 = PDEProblem(
        equation=equation,
        mesh=mesh,
        bc=bc,
        time_integrator=SSPRK3(),
        reconstruction=MinmodLinearReconstruction(),
    )

    print("Running Burgers equation (O1)...")
    result_o1 = solve(problem_o1, u0, tspan, label="cu1", state_labels=["u"])
    print("Running Burgers equation (O2)...")
    result_o2 = solve(problem_o2, u0, tspan, label="cu2", state_labels=["u"])

    plotter = SolutionPlotter(mesh)
    plotter.plot_solutions(
        [result_o1, result_o2],
        title=equation.name,
        n_time_stamps=5,
        t_max=t_max,
    )


def run_wave_animation():
    mesh = Mesh1D(x_left=-np.pi, x_right=np.pi, n_cells=200)
    bc = PeriodicBC()
    equation = WaveEquation1D(wave_speed=1.0)

    a1 = [-0.4, 1.0, -0.4]
    b1 = [-0.6, -0.8, 0.7]
    ux0 = trig_function(mesh.cell_centers, a=a1, b=b1, period=mesh.period)

    a2 = [0.1, 0.1, 0.7]
    b2 = [0.4, -0.7, -0.4]
    ut0 = trig_function(mesh.cell_centers, a=a2, b=b2, period=mesh.period)

    u0 = np.stack((ux0, ut0), axis=1)
    t_max = 2
    n_steps = 20
    tspan = np.linspace(0, t_max, n_steps + 1)

    problem_o1 = PDEProblem(
        equation=equation,
        mesh=mesh,
        bc=bc,
        time_integrator=ForwardEuler(),
        reconstruction=ConstantReconstruction(),
    )
    problem_o2 = PDEProblem(
        equation=equation,
        mesh=mesh,
        bc=bc,
        time_integrator=SSPRK3(),
        reconstruction=MinmodLinearReconstruction(),
    )

    print("Running wave animation (O1)...")
    result_o1 = solve(problem_o1, u0, tspan, label="uw1", state_labels=["ux", "ut"])
    print("Running wave animation (O2)...")
    result_o2 = solve(problem_o2, u0, tspan, label="uw2", state_labels=["ux", "ut"])

    true_sol_array = equation.get_true_solution(mesh, a1, b1, a2, b2, tspan)
    result_true = SolutionData("true", n_states=2, labels=["ux", "ut"], solution=true_sol_array)

    plotter = SolutionPlotter(mesh)
    plotter.plot_animation(
        [result_o1, result_o2, result_true],
        title=equation.name,
        t_max=t_max,
    )


if __name__ == "__main__":
    main_wave_equation()
    main_burgers_equation()
