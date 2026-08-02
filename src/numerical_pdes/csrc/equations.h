#pragma once
#include <torch/torch.h>
#include <string>
#include <tuple>

#include "mesh.h"
#include "boundary_conditions.h"
#include "reconstruction.h"

class Equation {
    private:
        std::string name_;
        int n_states_;

    public:
        Equation(std::string name, int n_states)
            : name_(name), n_states_(n_states) {}

        virtual ~Equation() = default;

        virtual double get_max_wave_speed(const torch::Tensor& u) = 0;

        virtual double compute_numerical_flux(
            const torch::Tensor& u, 
            const Mesh1d& mesh,
            const Reconstruction& reconstruction,
            const BoundaryCondition& boundary_condition,
            int n_ghost_cells) = 0;
        
        double compute_cfl_dt(const torch::Tensor& u, double dx, double cfl_number);
        
        std::tuple<double, double> eigenvalues(const torch::Tensor& u);

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
        get_local_speed(
            const torch::Tensor& u,
            const Mesh1d& mesh,
            const Reconstruction& reconstruction,
            const BoundaryCondition& boundary_condition,
            int n_ghost_cells
        );
};

