#include "time_integrators.h"
#include <iostream>


torch::Tensor TimeIntegrator::compute_rhs_1d(const torch::Tensor& u,
  const Equation& equation, 
  const Mesh1d& mesh, 
  const Reconstruction& reconstruction,
  const BoundaryCondition& boundary_condition,
  int n_ghost_cells) const {
    torch::Tensor flux = equation.compute_numerical_flux(
      u, 
      mesh,
      reconstruction,
      boundary_condition,
      n_ghost_cells);
    
    auto end = flux.size(1);
    return -(1.0 / mesh.GetDx()) * (flux.slice(1, 1, end) - flux.slice(1, 0, end - 1));
}


void ForwardEuler::step(
  torch::Tensor& u,
  double dt,
  const Equation& equation, 
  const Mesh1d& mesh, 
  const Reconstruction& reconstruction,
  const BoundaryCondition& boundary_condition,
  int n_ghost_cells) const {
    torch::Tensor u_middle = u.slice(1, n_ghost_cells, -n_ghost_cells);
    std::cout << "u_middle  = " << u_middle  << std::endl;
    u_middle  += dt * compute_rhs_1d(u, equation, mesh, reconstruction, boundary_condition, n_ghost_cells);
    boundary_condition.Apply(u, n_ghost_cells);
  }


void SSPRK3::step(
  torch::Tensor& u,
  double dt,
  const Equation& equation,
  const Mesh1d& mesh,
  const Reconstruction& reconstruction,
  const BoundaryCondition& boundary_condition,
  int n_ghost_cells) const {
    torch::Tensor u1 = torch::zeros_like(u);
    torch::Tensor u2 = torch::zeros_like(u);

    torch::Tensor rhs = compute_rhs_1d(u, equation, mesh, reconstruction, boundary_condition, n_ghost_cells);
    u1.slice(1, n_ghost_cells, -n_ghost_cells) = u.slice(1, n_ghost_cells, -n_ghost_cells) + dt * rhs;
    boundary_condition.Apply(u1, n_ghost_cells);

    rhs = compute_rhs_1d(u1, equation, mesh, reconstruction, boundary_condition, n_ghost_cells);
    u2.slice(1, n_ghost_cells, -n_ghost_cells) = 3.0/4.0 * u.slice(1, n_ghost_cells, -n_ghost_cells) + 1.0/4.0 * (u1.slice(1, n_ghost_cells, -n_ghost_cells) + dt * rhs);
    boundary_condition.Apply(u2, n_ghost_cells);

    rhs = compute_rhs_1d(u2, equation, mesh, reconstruction, boundary_condition, n_ghost_cells);
    u.slice(1, n_ghost_cells, -n_ghost_cells) = 1.0/3.0 * u.slice(1, n_ghost_cells, -n_ghost_cells) + 2.0/3.0 * (u2.slice(1, n_ghost_cells, -n_ghost_cells) + dt * rhs);
    boundary_condition.Apply(u, n_ghost_cells);
  }
