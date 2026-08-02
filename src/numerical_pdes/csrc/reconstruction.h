#pragma once
#include <torch/torch.h>
#include <tuple>

#include "mesh.h"
#include "boundary_conditions.h"

class Reconstruction {
    private:
        int required_ghost_cells_;

    public:
        Reconstruction(int required_ghost_cells)
            : required_ghost_cells_(required_ghost_cells) {}

        virtual ~Reconstruction() = default;

        virtual std::tuple<torch::Tensor, torch::Tensor> reconstruct(
            const torch::Tensor& u,
            const Mesh1d& mesh,
            const int n_ghost_cells,
            const BoundaryCondition& bc
        ) = 0;
};


class ConstantReconstruction : public Reconstruction {
    public:
        ConstantReconstruction(int required_ghost_cells = 1)
            : Reconstruction(required_ghost_cells) {} 

        std::tuple<torch::Tensor, torch::Tensor> reconstruct(
            const torch::Tensor& u,
            const Mesh1d& mesh,
            const int n_ghost_cells,
            const BoundaryCondition& bc
        ) override;
};
        


class MinmodLinearReconstruction : public Reconstruction {
    private:
        double theta_;

        // may add these two methods to a seperate num_pde math help namespace
        double _du_dx(double um1, double u, double up1, double dx);

        double _minmod(double a, double b, double c);

    public:
        MinmodLinearReconstruction(int required_ghost_cells = 1, double theta = 1.0)
            : Reconstruction(required_ghost_cells), theta_(theta) {} 

        std::tuple<torch::Tensor, torch::Tensor> reconstruct(
            const torch::Tensor& u,
            const Mesh1d& mesh,
            const int n_ghost_cells,
            const BoundaryCondition& bc
        ) override;
};
