#include "mesh.h"


double Mesh1d::GetPeriod() {
    return x_right_ - x_left_;
}


torch::Tensor Mesh1d::GetCellCentersWithGhostCells(int n_ghost_cells) const {
    double far_left = x_left_ - n_ghost_cells*dx_ + dx_ / 2.0;
    double far_right = x_right_ + n_ghost_cells*dx_ - dx_ / 2.0;
    int n_cells = cell_centers_.size(0);

    torch::Tensor x = torch::linspace(
                far_left,
                far_right,
                n_cells + 2 * n_ghost_cells,
                torch::kFloat64);
    return x;
}
