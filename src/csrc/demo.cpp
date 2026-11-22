#include <iostream>
#include <string>
#include <torch/torch.h>

#include "mesh.h"
#include "boundary_conditions.h"
#include "reconstruction.h"
#include "equations.h"
#include "time_integrators.h"

constexpr double pi = 3.14159265358979323846;

void demo_boundary_condition();
void demo_reconstruction();
void demo_mesh();
void demo_equations_BE();
void demo_equations_wave();
void run_time_integrator();
void run_wave_equation();

int main() {
  // demo_mesh();

  // demo_boundary_condition();

  // demo_reconstruction();

  // demo_equations_BE();

  // demo_equations_wave();

  run_time_integrator();

  // run_wave_equation();

  return 0;
}


void demo_mesh() {
  std::cout << "/////////////////" << std::endl;
  std::cout << "Demo Mesh" << std::endl;
  std::cout << "/////////////////" << std::endl;

  double lb = -pi;
  double rb = pi;
  int n_cells = 50;

  Mesh1d mesh = Mesh1d(lb, rb, n_cells);

  std::cout << "mesh.GetNCells() = " << mesh.GetNCells() << std::endl;

  std::cout << "mesh.GetPeriod() = " << mesh.GetPeriod() << std::endl;

  std::cout << "mesh.GetCellCenters() = " << mesh.GetCellCenters() << std::endl;
}

void demo_boundary_condition() {
  std::cout << "/////////////////" << std::endl;
  std::cout << "Demo Boundary Conditions" << std::endl;
  std::cout << "/////////////////" << std::endl;

  int n_cells = 6;
  int n_ghost_cells = 2;
  int n_states = 3;

  int n_rows = n_cells+2*n_ghost_cells;
  torch::Tensor u_p = torch::arange(n_rows, torch::TensorOptions().dtype(torch::kFloat64))
    .unsqueeze(0)
    .repeat({n_states, 1});

  std::cout << "u.size(0) = " << u_p.size(0) << std::endl;
  std::cout << "u.size(1) = " << u_p.size(1) << std::endl; 
  std::cout << "u.dim() = " << u_p.dim() << std::endl; 
  std::cout << "u = " << u_p << std::endl;

  PeriodicBC bc_p;

  bc_p.Apply(u_p, n_ghost_cells);

  std::cout << "u_p (after PeriodicBC - Apply) = " << u_p << std::endl;

  torch::Tensor u_left_p = torch::arange(n_cells+1, torch::TensorOptions().dtype(torch::kFloat64))
    .unsqueeze(0)
    .repeat({n_states, 1});
  torch::Tensor u_right_p = torch::arange(n_cells+1, torch::TensorOptions().dtype(torch::kFloat64))
    .unsqueeze(0)
    .repeat({n_states, 1});
  std::cout << "u_left_p = " << u_left_p << std::endl;
  std::cout << "u_right_p = " << u_right_p << std::endl;
  bc_p.ApplyInterface(u_right_p, u_left_p, n_cells);
  std::cout << "u_left_p (after PeriodicBC - ApplyInterface) = " << u_left_p << std::endl;
  std::cout << "u_right_p (after PeriodicBC - ApplyInterface) = " << u_right_p << std::endl;

  WallBC bc_w;

  torch::Tensor u_w = torch::arange(n_rows, torch::TensorOptions().dtype(torch::kFloat64))
    .unsqueeze(0)
    .repeat({n_states, 1});

  bc_w.Apply(u_w, n_ghost_cells);

  std::cout << "u_w (after WallBC - Apply) = " << u_w << std::endl;

  torch::Tensor u_left_w = torch::arange(n_cells+1, torch::TensorOptions().dtype(torch::kFloat64))
    .unsqueeze(0)
    .repeat({n_states, 1});
  torch::Tensor u_right_w = torch::arange(n_cells+1, torch::TensorOptions().dtype(torch::kFloat64))
    .unsqueeze(0)
    .repeat({n_states, 1});
  std::cout << "u_left_w = " << u_left_w << std::endl;
  std::cout << "u_right_w = " << u_right_w << std::endl;
  bc_w.ApplyInterface(u_right_w, u_left_w, n_cells);
  std::cout << "u_left_w (after WallBC - ApplyInterface) = " << u_left_w << std::endl;
  std::cout << "u_right_w (after WallBC - ApplyInterface) = " << u_right_w << std::endl;

  return;
}


void demo_reconstruction() {
  std::cout << "/////////////////" << std::endl;
  std::cout << "Demo Reconstruction" << std::endl;
  std::cout << "/////////////////" << std::endl;
    
  double lb = -pi;
  double rb = pi;
  int n_cells = 6;
  int n_ghost_cells = 2;
  int n_states = 3;

  double dx = (rb-lb) / n_cells;

  torch::Tensor x = torch::linspace(
    lb - n_ghost_cells*dx + dx / 2.0,
    rb + n_ghost_cells*dx - dx / 2.0,
    n_cells + 2 * n_ghost_cells,
    torch::kFloat64); 
  std::cout << "x.sizes() = " << x.sizes() << std::endl;
  std::cout << "x = " << x << std::endl;

  int n_rows = n_cells+2*n_ghost_cells;
  torch::Tensor u = torch::sin(x)
    .unsqueeze(0)
    .repeat({n_states, 1});

  std::cout << "u = " << u << std::endl;

  Mesh1d mesh = Mesh1d(lb, rb, n_cells);

  PeriodicBC bc;

  torch::Tensor u_left = torch::zeros(n_cells+1, torch::TensorOptions().dtype(torch::kFloat64))
    .unsqueeze(0)
    .repeat({n_states, 1});

  torch::Tensor u_right = torch::zeros(n_cells+1, torch::TensorOptions().dtype(torch::kFloat64))
    .unsqueeze(0)
    .repeat({n_states, 1});

  std::cout << "u_left (before const reconstruction) = " << u_left << std::endl;
  std::cout << "u_right (before const reconstruction) = " << u_right << std::endl;

  ConstantReconstruction const_rec = ConstantReconstruction();

  const_rec.reconstruct(
    u,
    mesh,
    n_ghost_cells,
    bc,
    u_right,
    u_left);

  std::cout << "u_left (after const reconstruction) = " << u_left << std::endl;
  std::cout << "u_right (after const reconstruction) = " << u_right << std::endl;

  u = torch::sin(x)
    .unsqueeze(0)
    .repeat({n_states, 1});

  MinmodLinearReconstruction linear_rec = MinmodLinearReconstruction();

  linear_rec.reconstruct(
    u,
    mesh,
    n_ghost_cells,
    bc,
    u_right,
    u_left);

  std::cout << "u_left (after linear reconstruction) = " << u_left << std::endl;
  std::cout << "u_right (after linear reconstruction) = " << u_right << std::endl;
}


void demo_equations_BE() {
  std::cout << "/////////////////" << std::endl;
  std::cout << "Demo Equations - Burgers Equation" << std::endl;
  std::cout << "/////////////////" << std::endl;
  double lb = -pi;
  double rb = pi;
  int n_cells = 6;
  int n_ghost_cells = 1;
  int n_states = 1;
  double dx = (rb-lb) / n_cells;

  torch::Tensor x = torch::linspace(
    lb - n_ghost_cells*dx + dx / 2.0,
    rb + n_ghost_cells*dx - dx / 2.0,
    n_cells + 2 * n_ghost_cells,
    torch::kFloat64); 

  std::cout << "x = " << x << std::endl;

  torch::Tensor u = torch::sin(x)
    .unsqueeze(0)
    .repeat({n_states, 1});

  std::cout << "u = " << u << std::endl;
    
  double diff_coef = 0.0;
  BurgersEquation1d b_equation = BurgersEquation1d(diff_coef);

  double max_wave_speed = b_equation.get_max_wave_speed(u);
  std::cout << "max_wave_speed = " << max_wave_speed << std::endl;

  double cfl_number = 0.5;
  double cfl_dt = b_equation.compute_cfl_dt(u, dx, cfl_number);
  std::cout << "cfl_dt = " << cfl_dt << std::endl;

  Mesh1d mesh = Mesh1d(lb, rb, n_cells);
  ConstantReconstruction reconstruction = ConstantReconstruction();
  PeriodicBC bc;
  torch::Tensor flux = b_equation.compute_numerical_flux(
    u, 
    mesh,
    reconstruction,
    bc,
    n_ghost_cells);
    
  std::cout << "u.select(/*dim=*/0, 0) = " << u.select(/*dim=*/0, 0) << std::endl;
  std::cout << "u.select(/*dim=*/1, 0) = " << u.select(/*dim=*/1, 0) << std::endl;
  std::cout << "u.size(0) = " << u.size(0) << std::endl;
  std::cout << "u.size(1) = " << u.size(1) << std::endl;
  std::cout << "flux = " << flux << std::endl;
}


void demo_equations_wave() {
  std::cout << "/////////////////" << std::endl;
  std::cout << "Demo Equations - Wave Equation" << std::endl;
  std::cout << "/////////////////" << std::endl;
  double lb = -pi;
  double rb = pi;
  int n_cells = 6;
  int n_ghost_cells = 1;
  int n_states = 2;
  double dx = (rb-lb) / n_cells;

  torch::Tensor x = torch::linspace(
    lb - n_ghost_cells*dx + dx / 2.0,
    rb + n_ghost_cells*dx - dx / 2.0,
    n_cells + 2 * n_ghost_cells,
    torch::kFloat64); 

  std::cout << "x = " << x << std::endl;

  torch::Tensor u = torch::sin(x)
    .unsqueeze(0)
    .repeat({n_states, 1});

  std::cout << "u = " << u << std::endl;
    
  double wave_speed = 2.0;
  WaveEquation1d wave_equation = WaveEquation1d(wave_speed);

  double max_wave_speed = wave_equation.get_max_wave_speed(u);
  std::cout << "max_wave_speed = " << max_wave_speed << std::endl;

  double cfl_number = 0.5;
  double cfl_dt = wave_equation.compute_cfl_dt(u, dx, cfl_number);
  std::cout << "cfl_dt = " << cfl_dt << std::endl;

  Mesh1d mesh = Mesh1d(lb, rb, n_cells);
  MinmodLinearReconstruction reconstruction = MinmodLinearReconstruction();
  PeriodicBC bc;
  torch::Tensor flux = wave_equation.compute_numerical_flux(
    u, 
    mesh,
    reconstruction,
    bc,
    n_ghost_cells);
    
  std::cout << "u.select(/*dim=*/0, 0) = " << u.select(/*dim=*/0, 0) << std::endl;
  std::cout << "u.select(/*dim=*/1, 0) = " << u.select(/*dim=*/1, 0) << std::endl;
  std::cout << "u.size(0) = " << u.size(0) << std::endl;
  std::cout << "u.size(1) = " << u.size(1) << std::endl;
  std::cout << "flux = " << flux << std::endl;
}


void run_time_integrator() {
  std::cout << "/////////////////" << std::endl;
  std::cout << "Demo Time Integrators" << std::endl;
  std::cout << "/////////////////" << std::endl;

  double lb = -pi;
  double rb = pi;
  int n_cells = 2;
  int n_ghost_cells = 1;
  int n_states = 2;
  double wave_speed = 1.0;
  double dx = (rb-lb) / n_cells;
  double dt = 0.1;

  Mesh1d mesh = Mesh1d(lb, rb, n_cells);
  PeriodicBC bc;

  torch::Tensor x = mesh.GetCellCentersWithGhostCells(n_ghost_cells);

  std::cout << "x = " << x << std::endl;

  torch::Tensor ux0 = torch::sin(mesh.GetCellCenters());
  torch::Tensor ut0 = torch::cos(mesh.GetCellCenters());

  std::cout << "ux0.dim() = " << ux0.dim() << std::endl;
  std::cout << "ux0.size(0) = " << ux0.size(0) << std::endl;

  torch::Tensor u = torch::zeros_like(x)
    .unsqueeze(0)
    .repeat({n_states, 1});

  // Forward Euler
  u.slice(1, n_ghost_cells, -n_ghost_cells)[0].copy_(ux0);
  u.slice(1, n_ghost_cells, -n_ghost_cells)[1].copy_(ut0);

  bc.Apply(u, n_ghost_cells);

  std::cout << "u.dim() = " << u.dim() << std::endl;
  std::cout << "u.size(0) = " << u.size(0) << std::endl;
  std::cout << "u.size(1) = " << u.size(1) << std::endl;

  std::cout << "u = " << u << std::endl;

  WaveEquation1d wave_equation = WaveEquation1d(wave_speed);
  ConstantReconstruction reconstruction = ConstantReconstruction();

  ForwardEuler fe_integrator = ForwardEuler();
  
  fe_integrator.step(u, dt, wave_equation, mesh, reconstruction, bc, n_ghost_cells);
  std::cout << "u (after ForwardEuler step) = " << u << std::endl;

  // SSPRK3
  double diff_coef = 0.0;
  n_states = 1;
  torch::Tensor u_be = torch::sin(x).unsqueeze(0);
  bc.Apply(u_be, n_ghost_cells);

  std::cout << "u_be.dim() = " << u_be.dim() << std::endl;
  std::cout << "u_be.size(0) = " << u_be.size(0) << std::endl;
  std::cout << "u_be = " << u_be << std::endl;

  BurgersEquation1d burgers_equation = BurgersEquation1d(diff_coef);
  MinmodLinearReconstruction linear_reconstruction = MinmodLinearReconstruction();

  SSPRK3 sspk3_integrator = SSPRK3();
  
  sspk3_integrator.step(u_be, dt, burgers_equation, mesh, linear_reconstruction, bc, n_ghost_cells);
  std::cout << "u (after SSPRK3 step) = " << u << std::endl;



}


void run_wave_equation() {
  int n_cells = 10;
  double lb = -pi;
  double rb = pi;
  double wave_speed = 1.0;
  int n_ghost_cells = 1;
  double t_max = 5.0;
  int n_steps = 20;


  Mesh1d mesh = Mesh1d(lb, rb, n_cells);
  BoundaryCondition* bc = new PeriodicBC();
  WaveEquation1d wave_equation = WaveEquation1d(wave_speed);

  torch::Tensor ux0 = torch::sin(mesh.GetCellCenters());
  torch::Tensor ut0 = torch::cos(mesh.GetCellCenters());

  std::cout << "ux0.dim() = " << ux0.dim() << std::endl;
  std::cout << "ux0.size(0) = " << ux0.size(0) << std::endl;

  torch::Tensor u0 = torch::stack({ux0, ut0}, 0);

  std::cout << "u0.dim() = " << u0.dim() << std::endl;
  std::cout << "u0.size(0) = " << u0.size(0) << std::endl;
  std::cout << "u0.size(1) = " << u0.size(1) << std::endl;


  torch::Tensor x = mesh.GetCellCentersWithGhostCells(n_ghost_cells); 

  std::cout << "x = " << x << std::endl;

  torch::Tensor tspan = torch::linspace(0, t_max, n_steps+1, torch::kFloat64);

  std::cout << "tspan = " << tspan << std::endl;
}