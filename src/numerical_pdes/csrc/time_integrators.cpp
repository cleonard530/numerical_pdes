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
    // auto start = n_ghost_cells;
    // auto end = u.size(1) - n_ghost_cells;
    // torch::Tensor u_middle = u.slice(1, start, end);


    // u_middle.add_(dt * rhs_fn(u));
    // boundary_condition.Apply(u, n_ghost_cells);
  }
