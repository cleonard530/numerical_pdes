#include <iostream>

#include "reconstruction.h"


void ConstantReconstruction::reconstruct(
    const torch::Tensor& u,
    const Mesh1d& mesh,
    const int n_ghost_cells,
    const BoundaryCondition& bc,
    torch::Tensor& u_right,
    torch::Tensor& u_left) const {
        int n_cells = mesh.GetNCells();
        int offset = n_ghost_cells - 1;

        u_right.copy_(u.slice(/*dim*/1, offset + 1, offset + n_cells + 2));
        u_left.copy_(u.slice(/*dim*/1, offset, offset + n_cells + 1));
        
        bc.ApplyInterface(u_right, u_left, n_cells);
    }


double MinmodLinearReconstruction::_du_dx(double um1, double u, double up1, double dx) const {
    double forward = theta_ * (up1 - u);
    double backward = theta_ * (u - um1);
    double central = (up1 - um1) / 2.0;

    return _minmod(forward, backward, central) / dx;
}


double MinmodLinearReconstruction::_minmod(double a, double b, double c) const {
    if ((a > 0) && (b > 0)) {
        return std::min({a, b, c});
    } 
    else if ((a < 0) && (b < 0)) {
        return std::max({a, b, c});
    }
    return 0;
}


void MinmodLinearReconstruction::reconstruct(
    const torch::Tensor& u,
    const Mesh1d& mesh,
    const int n_ghost_cells,
    const BoundaryCondition& bc,
    torch::Tensor& u_right,
    torch::Tensor& u_left) const {
        int n_cells = mesh.GetNCells();
        double dx = mesh.GetDx();
        int n_states = u.size(0);

        std::cout << "u_right.size(0) = " << u_right.size(0) << std::endl;
        std::cout << "u_left.size(0) = " << u_left.size(0) << std::endl;
        std::cout << "n_states = " << n_states << std::endl;
        TORCH_CHECK(u_right.size(0) == n_states, "u_right has the wrong number of states");
        TORCH_CHECK(u_left.size(0) == n_states, "u_left has the wrong number of states");
        TORCH_CHECK(u_right.size(1) == n_cells + 1, "u_right has the wrong number of cell interfaces");
        TORCH_CHECK(u_left.size(1) == n_cells + 1, "u_left has the wrong number of cell interfaces");

        int center = 0;
        double du = 0;
        for (int n = 0; n < n_states; n++) {
            for (int i = 0; i < n_cells; i++) {
                center = n_ghost_cells + i;
                du = _du_dx(
                    u[n][center - 1].item<double>(), 
                    u[n][center].item<double>(), 
                    u[n][center + 1].item<double>(), 
                    dx);
                u_left[n][i+1] = u[n][center].item<double>() + du * (dx / 2.0);
                u_right[n][i] = u[n][center].item<double>() - du * (dx / 2.0);
            }
        }
        
        bc.ApplyInterface(u_right, u_left, n_cells);
}