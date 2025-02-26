import numpy as np

from generate_data.base_pde_solver_1d import BasePDESolver1D


class WaveEquationSolver(BasePDESolver1D):
    def __init__(self, ax, bx, nx, wave_speed, bc='periodic'):
        super().__init__(ax, bx, nx, bc)
        self.wave_speed = wave_speed

    def get_solution(self, u0, tspan, method='cu2'):
        if method == 'uw1':
            self.order = 1
            u = self.get_uw1_solution(u0, tspan)
        elif method == 'uw2':
            self.order = 2
            u = self.get_uw2_solution(u0, tspan)
        else:
            assert False, f'method {method} is unrecognized.'
        return u

    def get_uw1_solution(self, u0, tspan):
        n_steps = len(tspan) - 1
        u = np.zeros([self.n_cells + 2, 2, n_steps + 1])

        u[1:self.n_cells + 1, :, 0] = u0
        self.set_ghost_cells(u[:, :, 0])

        for i in range(n_steps):
            time_step = tspan[i+1] - tspan[i]
            u[:, :, i+1] = self.get_next_step_o1(u[:, :, i], time_step)
            print(f"{tspan[i+1] =}")
        return u[1:self.n_cells + 1, :, :]

    def get_next_step_o1(self, u, time_step):
        t = 0
        dt = .5 * self.dx / self.wave_speed
        flux = np.zeros([self.n_cells + 1, 2])  # flux at cell interface

        while t < time_step - 1e-10:
            t += min(dt, time_step - t)

            flux[:, 0] = (1 / 2) * (self.wave_speed * u[:-1, 0] - u[:-1, 1] -
                                    self.wave_speed * u[1:, 0] - u[1:, 1])
            flux[:, 1] = (self.wave_speed / 2) * (-self.wave_speed * u[:-1, 0] + u[:-1, 1] -
                                                  self.wave_speed * u[1:, 0] - u[1:, 1])

            u[1:-1, 0] = u[1:-1, 0] - (dt / self.dx) * (flux[1:, 0] - flux[:-1, 0])
            u[1:-1, 1] = u[1:-1, 1] - (dt / self.dx) * (flux[1:, 1] - flux[:-1, 1])
            self.set_ghost_cells(u)

        return u

    def get_uw2_solution(self, u0, tspan):
        n_steps = len(tspan) - 1
        u = np.zeros([self.n_cells + 2, 2, n_steps + 1])

        u[1:self.n_cells + 1, :, 0] = u0
        self.set_ghost_cells(u[:, :, 0])

        for i in range(n_steps):
            time_step = tspan[i + 1] - tspan[i]
            u[:, :, i + 1] = self.get_next_step_o2(u[:, :, i], time_step)
            print(f"{tspan[i+1] =}")
        return u[1:self.n_cells + 1, :, :]

    def get_next_step_o2(self, u, time_step):
        t = 0
        dt = .5 * self.dx / self.wave_speed
        flux = np.zeros([self.n_cells + 1, 2])  # flux at cell interface

        u1 = np.zeros_like(u)
        u2 = np.zeros_like(u)
        while t < time_step - 1e-10:
            # first stage
            vp, vn = self.getLinear(u[:, 0], 1.0)
            wp, wn = self.getLinear(u[:, 1], 1.0)

            flux[:, 0] = (1 / 2) * (self.wave_speed * vn - wn - self.wave_speed * vp - wp)
            flux[:, 1] = (self.wave_speed / 2) * (-self.wave_speed * vn + wn - self.wave_speed * vp - wp)

            u1[1:-1, :] = u[1:-1, :] - (dt / self.dx) * (flux[1:, :] - flux[:-1, :])
            self.set_ghost_cells(u1)

            # second stage
            vp, vn = self.getLinear(u1[:, 0], 1.0)
            wp, wn = self.getLinear(u1[:, 1], 1.0)

            flux[:, 0] = (1 / 2) * (self.wave_speed * vn - wn - self.wave_speed * vp - wp)
            flux[:, 1] = (self.wave_speed / 2) * (-self.wave_speed * vn + wn - self.wave_speed * vp - wp)

            u2[1:-1, :] = .75*u[1:-1, :] + .25*(u1[1:-1, :] - (dt / self.dx) * (flux[1:, :] - flux[:-1, :]))
            self.set_ghost_cells(u2)

            # third stage
            vp, vn = self.getLinear(u2[:, 0], 1.0)
            wp, wn = self.getLinear(u2[:, 1], 1.0)

            flux[:, 0] = (1 / 2) * (self.wave_speed * vn - wn - self.wave_speed * vp - wp)
            flux[:, 1] = (self.wave_speed / 2) * (-self.wave_speed * vn + wn - self.wave_speed * vp - wp)

            u[1:-1, :] = (1/3.0) * u[1:-1, :] + (2/3.0) * (u2[1:-1, :] - (dt / self.dx) * (flux[1:, :] - flux[:-1, :]))
            self.set_ghost_cells(u2)

            t += min(dt, time_step - t)

        return u

    def getLinear(self, u, theta):
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
   nx = 200
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
   # tspan = np.linspace(0, .1, 5)
   u_uw1 = solver.get_solution(u0, tspan, method='uw1')
   u_uw2 = solver.get_solution(u0, tspan, method='uw2')

   solver.plot_solution([u_uw1[:, 0, :], u_uw1[:, 1, :], u_uw2[:, 0, :], u_uw2[:, 1, :]], t_max=t_max)

#
#
# def getSolution(a1, b1, a2, b2, c, n_steps, n_cells, k):
#    ve = np.zeros([n_steps+1, n_cells+2])
#    we = np.zeros([n_steps+1, n_cells+2])
#
#    l_boundary = -np.pi
#    r_boundary = np.pi
#
#    L = r_boundary-l_boundary
#    h = L/n_cells
#    x = np.linspace(l_boundary-(h/2),r_boundary+(h/2),n_cells+2)
#
#
#    ve[0,:] = dphi(x,a1,b1,k)
#    we[0,:] = phi(x,a2,b2,k)
#
#    t  = 0
#    dt = .5*h/c
#    for i in range(n_steps):
#       t  = t+dt
#       ve[i+1,:] = .5*(dphi(x-c*t,a1,b1,k)+dphi(x+c*t,a1,b1,k))+(1/(2*c))*(phi(x+c*t,a2,b2,k)-phi(x-c*t,a2,b2,k))
#       we[i+1,:] = (c/2)*(dphi(x+c*t,a1,b1,k)-dphi(x-c*t,a1,b1,k))+(1/2)*(phi(x+c*t,a2,b2,k)+phi(x-c*t,a2,b2,k))
#
#    return ve, we


if __name__ == '__main__':
   main()


