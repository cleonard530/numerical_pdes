import numpy as np

from generate_data.base_pde_solver_1d import BasePDESolver1D
from generate_data.solution_data import SolutionData

class WaveEquationSolver(BasePDESolver1D):
    def __init__(self, ax, bx, nx, n_states, wave_speed, bc='periodic'):
        super().__init__(ax, bx, nx, n_states, bc)
        self.wave_speed = wave_speed

    def get_solution(self, u0, tspan, method='cu2'):
        if method == 'uw1':
            self.n_ghost_cells = 1
            u = self.run_solver(u0, tspan, self.get_uw_next_step_o1, self.get_uw_flux_o1)
        elif method == 'uw2':
            self.n_ghost_cells = 1
            u = self.run_solver(u0, tspan, self.get_ssp_rk3_next_step, self.get_uw_flux_o2)
        else:
            assert False, f'method {method} is unrecognized.'
        return u

    def get_uw_next_step_o1(self, u, time_step, flux_function):
        t = 0
        dt = .5 * self.dx / self.wave_speed

        while t < time_step - 1e-10:
            t += min(dt, time_step - t)

            flux = flux_function(u)  # self.get_uw_flux_o1(u)
            u[1:-1, 0] = u[1:-1, 0] - (dt / self.dx) * (flux[1:, 0] - flux[:-1, 0])
            u[1:-1, 1] = u[1:-1, 1] - (dt / self.dx) * (flux[1:, 1] - flux[:-1, 1])
            self.set_ghost_cells(u)

        return u

    def get_ssp_rk3_next_step(self, u, time_step, flux_function):
        t = 0
        dt = .5 * self.dx / self.wave_speed

        u1 = np.zeros_like(u)
        u2 = np.zeros_like(u)
        while t < time_step - 1e-10:
            # first stage
            flux = flux_function(u)  #self.get_uw_flux_o2(u)
            u1[1:-1, :] = u[1:-1, :] - (dt / self.dx) * (flux[1:, :] - flux[:-1, :])
            self.set_ghost_cells(u1)

            # second stage
            flux = flux_function(u1)
            u2[1:-1, :] = .75*u[1:-1, :] + .25*(u1[1:-1, :] - (dt / self.dx) * (flux[1:, :] - flux[:-1, :]))
            self.set_ghost_cells(u2)

            # third stage
            flux = flux_function(u2)
            u[1:-1, :] = (1/3.0) * u[1:-1, :] + (2/3.0) * (u2[1:-1, :] - (dt / self.dx) * (flux[1:, :] - flux[:-1, :]))
            self.set_ghost_cells(u)

            t += min(dt, time_step - t)

        return u

    def get_cu_next_step_o2(self, u, time_step):
        t = 0
        dt = .5 * self.dx / self.wave_speed

        u1 = np.zeros_like(u)
        u2 = np.zeros_like(u)
        while t < time_step - 1e-10:
            # first stage
            flux = self.get_uw_flux_o2(u)
            u1[1:-1, :] = u[1:-1, :] - (dt / self.dx) * (flux[1:, :] - flux[:-1, :])
            self.set_ghost_cells(u1)

            # second stage
            flux = self.get_uw_flux_o2(u1)
            u2[1:-1, :] = .75*u[1:-1, :] + .25*(u1[1:-1, :] - (dt / self.dx) * (flux[1:, :] - flux[:-1, :]))
            self.set_ghost_cells(u2)

            # third stage
            flux = self.get_uw_flux_o2(u2)
            u[1:-1, :] = (1/3.0) * u[1:-1, :] + (2/3.0) * (u2[1:-1, :] - (dt / self.dx) * (flux[1:, :] - flux[:-1, :]))
            self.set_ghost_cells(u)

            t += min(dt, time_step - t)

        return u

    def get_uw_flux_o1(self, u):
        flux = np.zeros([self.n_cells + 1, 2])  # flux at cell interface

        flux[:, 0] = (1 / 2) * (self.wave_speed * u[:-1, 0] - u[:-1, 1] -
                                self.wave_speed * u[1:, 0] - u[1:, 1])
        flux[:, 1] = (self.wave_speed / 2) * (-self.wave_speed * u[:-1, 0] + u[:-1, 1] -
                                              self.wave_speed * u[1:, 0] - u[1:, 1])
        return flux

    def get_uw_flux_o2(self, u):
        flux = np.zeros([self.n_cells + 1, 2])  # flux at cell interface

        vp, vn = self.get_linear_cell_boundary_approximation(u[:, 0], 1.0)
        wp, wn = self.get_linear_cell_boundary_approximation(u[:, 1], 1.0)

        flux[:, 0] = (1 / 2) * (self.wave_speed * vn - wn - self.wave_speed * vp - wp)
        flux[:, 1] = (self.wave_speed / 2) * (-self.wave_speed * vn + wn - self.wave_speed * vp - wp)

        return flux

    def get_cu2_flux_o2(self, u):
        flux = np.zeros([self.n_cells + 1, 2])  # flux at cell interface

        vp, vn = self.get_linear_cell_boundary_approximation(u[:, 0], 1.0)
        wp, wn = self.get_linear_cell_boundary_approximation(u[:, 1], 1.0)

        flux[:, 0] = (1 / 2) * (self.wave_speed * vn - wn - self.wave_speed * vp - wp)
        flux[:, 1] = (self.wave_speed / 2) * (-self.wave_speed * vn + wn - self.wave_speed * vp - wp)

        return flux

def main():
    ax = -np.pi
    bx = np.pi
    nx = 100
    wave_speed = 1

    solver = WaveEquationSolver(ax=ax,
                                bx=bx,
                                nx=nx,
                                n_states=2,
                                wave_speed=wave_speed)

    v0 = solver.trig_initial_condition()
    w0 = solver.trig_initial_condition()

    t_max = 1
    n_steps = 10
    tspan = np.linspace(0, t_max, n_steps+1)
    u0 = np.stack((v0, w0), axis=1)
    solutions = [SolutionData('uw1', labels=['ux', 'ut']),
                 SolutionData('uw2', labels=['ux', 'ut'])]    #[{'method': 'uw1', 'labels': ['ux', 'ut']},
                    #{'method': 'uw2', 'labels': ['ux', 'ut']}]

    for sol in solutions:
        sol.set_solution(solver.get_solution(u0, tspan, method=sol.method))

    solver.plot_animation(solutions, t_max, equation='Wave Equation')
    # solver.plot_solution(solutions, t_max=t_max)


if __name__ == '__main__':
   main()
