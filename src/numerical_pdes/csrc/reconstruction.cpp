#include "reconstruction.h"


std::tuple<torch::Tensor, torch::Tensor> ConstantReconstruction::reconstruct(
    const torch::Tensor& u,
    const Mesh1d& mesh,
    const int n_ghost_cells,
    const BoundaryCondition& bc) {
        int n_cells = mesh.GetNCells();
        int offset = n_ghost_cells - 1;

        torch::Tensor u_right = u.slice(/*dim*/0, offset, offset + n_cells + 1).clone();
        torch::Tensor u_left = u.slice(/*dim*/0, offset + 1, offset + n_cells + 2).clone();
        
        bc.ApplyInterface(u_right, u_left, n_cells);
        return {u_right, u_left};
}


double MinmodLinearReconstruction::_du_dx(double um1, double u, double up1, double dx) {
    double forward = theta_ * (up1 - u);
    double backward = theta_ * (u - um1);
    double central = (up1 - um1) / 2.0;

    return _minmod(forward, backward, central) / dx;
}


double MinmodLinearReconstruction::_minmod(double a, double b, double c) {
    if ((a > 0) && (b > 0)) {
        return std::min({a, b, c});
    } 
    else if ((a < 0) && (b < 0)) {
        return std::max({a, b, c});
    }
    return 0;
}


std::tuple<torch::Tensor, torch::Tensor> MinmodLinearReconstruction::reconstruct(
    const torch::Tensor& u,
    const Mesh1d& mesh,
    const int n_ghost_cells,
    const BoundaryCondition& bc) {
        int n_cells = mesh.GetNCells();
        double dx = mesh.GetDx();

        torch::Tensor u_right = torch::zeros({n_cells + 1}, u.options());
        torch::Tensor u_left = torch::zeros({n_cells + 1}, u.options());
        
        int center = 0;
        double du = 0;
        for (int i = 0; i < n_cells; i++) {
            center = n_ghost_cells + i;
            du = _du_dx(
                u[center - 1].item<double>(), 
                u[center].item<double>(), 
                u[center + 1].item<double>(), 
                dx);
            u_left[i+1] = u[center].item<double>() + du * (dx / 2.0);
            u_right[i] = u[center].item<double>() - du * (dx / 2.0);
        }
        
        bc.ApplyInterface(u_right, u_left, n_cells);
        return {u_right, u_left};
}