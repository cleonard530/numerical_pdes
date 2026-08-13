#include <iostream>
#include <string>
#include <torch/torch.h>

#include "mesh.h"
#include "boundary_conditions.h"
#include "reconstruction.h"
#include "equations.h"

constexpr double pi = 3.14159265358979323846;

void demo_boundary_condition();


int main() {
    double lb = -pi;
    double rb = pi;
    int n_cells = 50;

    std::cout << rb - lb << std::endl;

    Mesh1d mesh = Mesh1d(lb, rb, n_cells);

    std::cout << mesh.GetNCells() << std::endl;

    demo_boundary_condition();

    return 0;
}


void demo_boundary_condition() {
    int n_cells = 50;
    int n_ghost_cells = 2;
    int n_states = 3;

    int n_rows = n_cells+2*n_ghost_cells;
    torch::Tensor u = torch::arange(n_rows, torch::TensorOptions().dtype(torch::kFloat64))
        .unsqueeze(0)
        .repeat({n_states, 1});

    std::cout << "u.size(0) = " << u.size(0) << std::endl;
    std::cout << "u.size(1) = " << u.size(1) << std::endl; 
    std::cout << "u.dim() = " << u.dim() << std::endl; 
    std::cout << "u = " << u << std::endl;

    WallBC bc;

    bc.Apply(u, n_ghost_cells);

    std::cout << "u = " << u << std::endl;

    torch::Tensor u_left = torch::arange(n_cells+1, torch::TensorOptions().dtype(torch::kFloat64));
    torch::Tensor u_right = torch::arange(n_cells+1, torch::TensorOptions().dtype(torch::kFloat64));
    std::cout << "u_left0 = " << u_left << std::endl;
    std::cout << "u_right0 = " << u_right << std::endl;
    bc.ApplyInterface(u_right, u_left, n_cells);
    std::cout << "u_left1 = " << u_left << std::endl;
    std::cout << "u_right1 = " << u_right << std::endl;

    return;
}


