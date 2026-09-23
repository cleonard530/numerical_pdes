#include "equations.h"
#include <cmath>

// Equation

std::tuple<double, double> Equation::get_eigenvalues(const torch::Tensor& u) { return {0, 0}; }

double Equation::compute_cfl_dt(const torch::Tensor& u, double dx, double cfl_number) {
  double speed = get_max_wave_speed(u);
  if (speed < 1e-14) {
    return dx;
  }
  return cfl_number * dx / speed;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Equation::get_local_speed(
    const torch::Tensor& u,
    const Mesh1d& mesh,
    const Reconstruction& reconstruction,
    const BoundaryCondition& boundary_condition,
    int n_ghost_cells) {
  int n_cells = mesh.GetNCells();

  torch::Tensor a_negative = torch::zeros({n_cells + 1}, u.options());
  torch::Tensor a_positive = torch::zeros({n_cells + 1}, u.options());
  torch::Tensor u_right = torch::zeros({n_states_, n_cells + 1}, u.options());
  torch::Tensor u_left = torch::zeros({n_states_, n_cells + 1}, u.options());

  reconstruction.reconstruct(u, mesh, n_ghost_cells, boundary_condition, u_right, u_left);

  for (int i = 0; i < n_cells + 1; i++) {
    auto [eig_max_l, eig_min_l] = get_eigenvalues(u_left.select(/*dim=*/1, i));
    auto [eig_max_r, eig_min_r] = get_eigenvalues(u_right.select(/*dim=*/1, i));
    a_positive[i] = std::max({eig_max_l, eig_max_r, 0.0});
    a_negative[i] = std::min({eig_min_l, eig_min_r, 0.0});
  }
  return {a_negative, a_positive, u_right, u_left};
}


// Burgers Equation

bool BurgersEquation1d::_is_viscous(double tol) {
  if (diff_coef_ > tol) { return true; }
  else if (diff_coef_ >= 0) { return false; }
  TORCH_CHECK(false,
    "diffusion coefficient (" + std::to_string(diff_coef_) + ") must be non-negative");
}


double BurgersEquation1d::get_max_wave_speed(const torch::Tensor& u) {
  return torch::max(torch::abs(u)).item<double>();
}


std::tuple<double, double> BurgersEquation1d::get_eigenvalues(const torch::Tensor& u) {
  double u0 = u[0].item<double>();
  return {u0, u0};
}


double BurgersEquation1d::compute_cfl_dt(const torch::Tensor& u, double dx, double cfl_number) {
  double dt = Equation::compute_cfl_dt(u, dx, cfl_number);
  if (_is_viscous()) {
      dt = std::min(dt, cfl_number * dx * dx / diff_coef_);
  }
  return dt;
}


torch::Tensor BurgersEquation1d::_get_physical_flux(torch::Tensor u) {
  return torch::square(u) / 2.0;
}


torch::Tensor BurgersEquation1d::compute_numerical_flux(
    const torch::Tensor& u, 
    const Mesh1d& mesh,
    const Reconstruction& reconstruction,
    const BoundaryCondition& boundary_condition,
    int n_ghost_cells) {
  int n_cells = mesh.GetNCells();
  torch::Tensor flux = torch::zeros({n_states_, n_cells + 1});

  auto [a_neg, a_pos, u_right, u_left] = get_local_speed(
    u,
    mesh,
    reconstruction,
    boundary_condition,
    n_ghost_cells
  );

  torch::Tensor f_left = _get_physical_flux(u_left);
  torch::Tensor f_right = _get_physical_flux(u_right);

  for (int i = 0; i < n_cells + 1; i++) {
    if ((a_pos[i] - a_neg[i]).item<double>() > 1e-8) {
      flux[0][i] = (
        a_pos[i] * f_left[0][i]
        - a_neg[i] * f_right[0][i]
        + a_pos[i] * a_neg[i] * (u_right[0][i] - u_left[0][i])
      ) / (a_pos[i] - a_neg[i]);
    }
    else {
      flux[0][i] = (f_left[0][i] + f_right[0][i]) / 2;
    }
  }

  return flux;
}


// Wave Equation

double WaveEquation1d::get_max_wave_speed(const torch::Tensor& u) {
  return wave_speed_;
}


std::tuple<double, double> WaveEquation1d::get_eigenvalues(const torch::Tensor& u) {
  double c = std::sqrt(wave_speed_);
  return {c, -c};
}


torch::Tensor WaveEquation1d::compute_numerical_flux(
    const torch::Tensor& u, 
    const Mesh1d& mesh,
    const Reconstruction& reconstruction,
    const BoundaryCondition& boundary_condition,
    int n_ghost_cells) {
  int n_cells = mesh.GetNCells();
  torch::Tensor flux = torch::zeros({n_states_, n_cells + 1});

  torch::Tensor u_right = torch::zeros({n_states_, n_cells + 1}, u.options());
  torch::Tensor u_left = torch::zeros({n_states_, n_cells + 1}, u.options());

  reconstruction.reconstruct(u, mesh, n_ghost_cells, boundary_condition, u_right, u_left);
  torch::Tensor vp = u_right.select(/*dim=*/0, 0);
  torch::Tensor vn = u_left.select(/*dim=*/0, 0);
  torch::Tensor wp = u_right.select(/*dim=*/0, 1);
  torch::Tensor wn = u_left.select(/*dim=*/0, 1);

  double c = wave_speed_;
  flux.select(/*dim=*/0, 0) = 0.5 * (c * vn - wn - c * vp - wp);
  flux.select(/*dim=*/0, 1) = c * 0.5 * (-c * vn + wn - c * vp - wp);  
      
  return flux;
}