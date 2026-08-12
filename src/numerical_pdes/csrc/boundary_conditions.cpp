#include "boundary_conditions.h"


void PeriodicBC::Apply(torch::Tensor& u, int n_ghost_cells) {
    int n_cells = u.size(1) - 2 * n_ghost_cells;
    for (int i=0; i < n_ghost_cells; i++) {
        u.select(/*dim=*/1, i).copy_(u.select(/*dim=*/1, i + n_cells));
        u.select(/*dim=*/1, n_ghost_cells + n_cells + i).copy_(u.select(/*dim=*/1, n_ghost_cells + i));
    }
}


void PeriodicBC::ApplyInterface(torch::Tensor& u_right, torch::Tensor& u_left, int n_cells) {
    u_left[0] = u_left[n_cells];
    u_right[n_cells] = u_right[0];
}


void WallBC::Apply(torch::Tensor& u, int n_ghost_cells) {
    int n_cells = u.size(1) - 2 * n_ghost_cells;
    for (int i=0; i < n_ghost_cells; i++) {
        u.select(/*dim=*/1, i).copy_(u.select(/*dim=*/1, 2 * n_ghost_cells - 1 - i));
        u.select(/*dim=*/1, n_ghost_cells + n_cells + i).copy_(u.select(/*dim=*/1, n_ghost_cells + n_cells - 1 - i));
    }
}


void WallBC::ApplyInterface(torch::Tensor& u_right, torch::Tensor& u_left, int n_cells) {
    u_left[0] = u_right[0];
    u_right[n_cells] = u_left[n_cells];
}

