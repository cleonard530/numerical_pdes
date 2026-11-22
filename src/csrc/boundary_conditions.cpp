#include "boundary_conditions.h"


void PeriodicBC::Apply(torch::Tensor& u, int n_ghost_cells) const {
  int n_cells = u.size(1) - 2 * n_ghost_cells;
  for (int i=0; i < n_ghost_cells; i++) {
    u.select(/*dim=*/1, i).copy_(u.select(/*dim=*/1, i + n_cells));
    u.select(/*dim=*/1, n_ghost_cells + n_cells + i).copy_(u.select(/*dim=*/1, n_ghost_cells + i));
  }
}


void PeriodicBC::ApplyInterface(torch::Tensor& u_right, torch::Tensor& u_left, int n_cells) const {
  u_left.select(/*dim=*/1, 0).copy_(u_left.select(/*dim=*/1, n_cells));
  u_right.select(/*dim=*/1, n_cells).copy_(u_right.select(/*dim=*/1, 0));
}


void WallBC::Apply(torch::Tensor& u, int n_ghost_cells) const {
  int n_cells = u.size(1) - 2 * n_ghost_cells;
  for (int i=0; i < n_ghost_cells; i++) {
    u.select(/*dim=*/1, i).copy_(u.select(/*dim=*/1, 2 * n_ghost_cells - 1 - i));
    u.select(/*dim=*/1, n_ghost_cells + n_cells + i).copy_(u.select(/*dim=*/1, n_ghost_cells + n_cells - 1 - i));
  }
}


void WallBC::ApplyInterface(torch::Tensor& u_right, torch::Tensor& u_left, int n_cells) const {
  u_left.select(/*dim=*/1, 0).copy_(u_right.select(/*dim=*/1, 0));
  u_right.select(/*dim=*/1, n_cells).copy_(u_left.select(/*dim=*/1, n_cells));
}
