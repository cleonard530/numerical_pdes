import numpy as np

from generate_data.base_pde_solver_1d import BasePDESolver1D


class WaveEquationSolver(BasePDESolver1D):
    def __init__(self, ax, bx, nx, wave_speed, bc='periodic'):
        super().__init__(ax, bx, nx, bc)
        self.wave_speed = wave_speed

    def get_solution(self, u0, tspan, method='cu2'):
        if method == 'uw1':
            self.order = 1
            u = self.run_solver(u0, tspan, self.get_uw_next_step_o1)
        elif method == 'uw2':
            self.order = 1
            u = self.run_solver(u0, tspan, self.get_uw_next_step_o2)
        else:
            assert False, f'method {method} is unrecognized.'
        return u

    def get_uw_next_step_o1(self, u, time_step):
        t = 0
        dt = .5 * self.dx / self.wave_speed

        while t < time_step - 1e-10:
            t += min(dt, time_step - t)

            flux = self.get_uw_flux_o1(u)
            u[1:-1, 0] = u[1:-1, 0] - (dt / self.dx) * (flux[1:, 0] - flux[:-1, 0])
            u[1:-1, 1] = u[1:-1, 1] - (dt / self.dx) * (flux[1:, 1] - flux[:-1, 1])
            self.set_ghost_cells(u)

        return u

    def get_uw_next_step_o2(self, u, time_step):
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

        vp, vn = self.get_linear(u[:, 0], 1.0)
        wp, wn = self.get_linear(u[:, 1], 1.0)

        flux[:, 0] = (1 / 2) * (self.wave_speed * vn - wn - self.wave_speed * vp - wp)
        flux[:, 1] = (self.wave_speed / 2) * (-self.wave_speed * vn + wn - self.wave_speed * vp - wp)

        return flux

    def get_linear(self, u, theta):
        up = np.zeros([self.n_cells + 1])
        un = np.zeros([self.n_cells + 1])

        for i in range(self.n_cells):
            du = self.dudx(u[i], u[i + 1], u[i + 2], self.dx, theta)
            un[i + 1] = u[i + 1] + du * (self.dx / 2)
            up[i] = u[i + 1] - du * (self.dx / 2)
        un[0] = un[self.n_cells]
        up[self.n_cells] = up[0]

        return up, un

    def trig_initial_condition(self, a=None, b=None, k=3):
       x = np.linspace(self.ax + (self.dx / 2), self.bx - (self.dx / 2), self.n_cells)
       if a is None:
           a = np.random.uniform(-1, 1, k)
       if b is None:
           b = np.random.uniform(-1, 1, k)
       period = self.bx - self.ax

       y = np.zeros(x.shape)
       for i in range(1, k + 1):
          y = y + (1 / i) * (a[i - 1] * np.cos(2*np.pi*i*x/period) +
                             b[i - 1] * np.sin(2*np.pi*i*x/period))
       return y


def main():
   ax = -np.pi
   bx = np.pi
   nx = 100
   wave_speed = 1

   solver = WaveEquationSolver(ax=ax,
                               bx=bx,
                               nx=nx,
                               wave_speed=wave_speed)

   v0 = solver.trig_initial_condition()
   w0 = solver.trig_initial_condition()

   t_max = 10
   tspan = np.linspace(0, t_max, 101)
   u0 = np.stack((v0, w0), axis=1)
   u_uw1 = solver.get_solution(u0, tspan, method='uw1')
   u_uw2 = solver.get_solution(u0, tspan, method='uw2')

   solver.plot_solution([u_uw1[:, 0, :], u_uw1[:, 1, :], u_uw2[:, 0, :], u_uw2[:, 1, :]], t_max=t_max)


if __name__ == '__main__':
   main()
