#include "mesh.h"


double Mesh1d::GetPeriod() {
    return x_right_ - x_left_;
}


torch::Tensor Mesh1d::GetCellCentersWithGhostCells(int n_ghost_cells) const {
    auto n_cells = cell_centers_.size(0);
    auto far_left = cell_centers_.index({0}) - n_ghost_cells*dx_;
    auto far_right = cell_centers_.index({n_cells - 1}) + n_ghost_cells*dx_;

    torch::Tensor x = torch::linspace(
                far_left,
                far_right,
                n_cells + 2 * n_ghost_cells,
                torch::kFloat64);
    return x;
}
