#pragma once
#include <torch/torch.h>


class BoundaryCondition {
    public:    
        // Virtual destructor
        virtual ~BoundaryCondition() = default;
        
        virtual void Apply(torch::Tensor& u, int n_ghost_cells) const = 0;

        virtual void ApplyInterface(torch::Tensor& u_right, torch::Tensor& u_left, int n_cells) const = 0;        
};


class PeriodicBC : public BoundaryCondition {
    public:
        void Apply(torch::Tensor& u, int n_ghost_cells) const override;

        void ApplyInterface(torch::Tensor& u_right, torch::Tensor& u_left, int n_cells) const override;
};


class WallBC : public BoundaryCondition {
    public:
        void Apply(torch::Tensor& u, int n_ghost_cells) const override;

        void ApplyInterface(torch::Tensor& u_right, torch::Tensor& u_left, int n_cells) const override;
};