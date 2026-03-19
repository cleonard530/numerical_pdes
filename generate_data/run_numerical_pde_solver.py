from equation_solvers import WaveEquationSolver, BurgersEquationSolver
from solution_data import SolutionData
import numpy as np

from utils.functions import trig_function, piecewise_constant
from utils import errors
from utils.plot_solutions import SolutionPlotter


def main_wave_equation():
    ax = -np.pi
    bx = np.pi
    nx = 200
    wave_speed = 1

    solver = WaveEquationSolver(ax=ax,
                                bx=bx,
                                n_cells=nx,
                                wave_speed=wave_speed)

    a1 = [-0.4, 1.0, -0.4]
    b1 = [-0.6, -0.8, 0.7]
    ux0 = trig_function(solver.x, a=a1, b=b1, period=bx-ax)

    a2 = [0.1, 0.1, 0.7]
    b2 = [0.4, -0.7, -0.4]
    ut0 = trig_function(solver.x, a=a2, b=b2, period=bx-ax)

    t_max = 5
    n_steps = 20
    tspan = np.linspace(0, t_max, n_steps+1)
    u0 = np.stack((ux0, ut0), axis=1)

    solution_data = [SolutionData('uw1', n_states=2, labels=['ux', 'ut']),
                     SolutionData('uw2', n_states=2, labels=['ux', 'ut'])]
    for sol in solution_data:
        sol.set_solution(solver.get_solution(u0, tspan, method=sol.method))

    solution_data.append(solver.get_true_solution(a1, b1, a2, b2, tspan))

    ux1_rel_error2 = errors.l2_error(solution_data[0].solution[:, 0, -1], solution_data[2].solution[:, 0, -1])
    ut1_rel_error2 = errors.l2_error(solution_data[0].solution[:, 1, -1], solution_data[2].solution[:, 1, -1])
    ux2_rel_error2 = errors.l2_error(solution_data[1].solution[:, 0, -1], solution_data[2].solution[:, 0, -1])
    ut2_rel_error2 = errors.l2_error(solution_data[1].solution[:, 1, -1], solution_data[2].solution[:, 1, -1])

    print(f"{ux1_rel_error2 = }")
    print(f"{ut1_rel_error2 = }")
    print(f"{ux2_rel_error2 = }")
    print(f"{ut2_rel_error2 = }")
    solution_plotter = SolutionPlotter(ax=ax, bx=bx)
    solution_plotter.plot_solutions(solution_data, title=solver.equation, n_times_stamps=4, t_max=t_max)


def main_burgers_equation():
    ax = -np.pi
    bx = np.pi
    nx = 500
    diff_coef = 0

    solver = BurgersEquationSolver(ax=ax,
                                   bx=bx,
                                   n_cells=nx,
                                   diff_coef=diff_coef,
                                   bc='wall')
    u_left = -0.2
    u_right = 0.6
    u0 = np.zeros((nx, 1))
    u0[:, 0] = piecewise_constant(nx, u_left=u_left, u_right=u_right)

    t_max = 1
    n_steps = 10
    tspan = np.linspace(0, t_max, n_steps + 1)

    solution_data = [SolutionData('cu1', n_states=1, labels=['u']), SolutionData('cu2', n_states=1, labels=['u'])]
    for sol in solution_data:
        sol.set_solution(solver.get_solution(u0, tspan, method=sol.method))

    solution_plotter = SolutionPlotter(ax=ax, bx=bx)
    solution_plotter.plot_solutions(solution_data, title=solver.equation, n_times_stamps=5, t_max=t_max)


def run_plot_animation():
    ax = -np.pi
    bx = np.pi
    nx = 200
    wave_speed = 1

    solver = WaveEquationSolver(ax=ax,
                                bx=bx,
                                n_cells=nx,
                                wave_speed=wave_speed)

    a1 = [-0.4, 1.0, -0.4]
    b1 = [-0.6, -0.8, 0.7]
    ux0 = trig_function(solver.x, a=a1, b=b1, period=bx - ax)

    a2 = [0.1, 0.1, 0.7]
    b2 = [0.4, -0.7, -0.4]
    ut0 = trig_function(solver.x, a=a2, b=b2, period=bx - ax)

    u0 = np.stack((ux0, ut0), axis=1)

    t_max = 2
    n_steps = 20
    tspan = np.linspace(0, t_max, n_steps+1)
    solution_data = [SolutionData('uw1', n_states=2, labels=['ux', 'ut']),
                     SolutionData('uw2', n_states=2, labels=['ux', 'ut'])]
    for sol in solution_data:
        sol.set_solution(solver.get_solution(u0, tspan, method=sol.method))

    solution_data.append(solver.get_true_solution(a1, b1, a2, b2, tspan))

    solution_plotter = SolutionPlotter(ax=ax, bx=bx)
    solution_plotter.plot_animation(solution_data, title=solver.equation, t_max=t_max)


def run_animation2():
    ax = -np.pi
    bx = np.pi
    nx = 100
    wave_speed = 1

    solver = WaveEquationSolver(ax=ax,
                                bx=bx,
                                n_cells=nx,
                                wave_speed=wave_speed)

    t_max = 2
    n_steps = 100
    tspan = np.linspace(0, t_max, n_steps + 1)
    a1 = [-0.4, 1.0, -0.4]
    b1 = [-0.6, -0.8, 0.7]
    ux0 = trig_function(solver.x, a=a1, b=b1, period=bx - ax)

    a2 = [0.1, 0.1, 0.7]
    b2 = [0.4, -0.7, -0.4]
    ut0 = trig_function(solver.x, a=a2, b=b2, period=bx - ax)
    u0 = np.stack((ux0, ut0), axis=1)

    solution_data = [SolutionData('uw1', n_states=2, labels=['ux', 'ut']),
                     SolutionData('uw2', n_states=2, labels=['ux', 'ut'])]

    solution_plotter = SolutionPlotter(ax=ax, bx=bx)
    solution_plotter.run_with_animator(solver, u0, solution_data, t_span=tspan)


if __name__ == '__main__':
    # main_wave_equation()
    # main_burgers_equation()
    run_plot_animation()
    #run_animation2()
