#pragma once
#include <torch/torch.h>
#include <string>
#include <tuple>

#include "mesh.h"
#include "boundary_conditions.h"
#include "reconstruction.h"

class Equation {
    protected:
        std::string name_;
        int n_states_;

    public:
        Equation(std::string name, int n_states)
            : name_(name), n_states_(n_states) {}

        virtual ~Equation() = default;

        virtual double get_max_wave_speed(const torch::Tensor& u) = 0;

        virtual torch::Tensor compute_numerical_flux(
            const torch::Tensor& u, 
            const Mesh1d& mesh,
            const Reconstruction& reconstruction,
            const BoundaryCondition& boundary_condition,
            int n_ghost_cells) = 0;
        
        virtual std::tuple<double, double> get_eigenvalues(const torch::Tensor& u);

        virtual double compute_cfl_dt(const torch::Tensor& u, double dx, double cfl_number);

        virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
        get_local_speed(
            const torch::Tensor& u,
            const Mesh1d& mesh,
            const Reconstruction& reconstruction,
            const BoundaryCondition& boundary_condition,
            int n_ghost_cells
        );
};



class BurgersEquation1d : public Equation {
    private:
        double diff_coef_;

        bool _is_viscous(double tol=1e-8);

        torch::Tensor _get_physical_flux(torch::Tensor u);

    public:
        BurgersEquation1d(double diff_coef)
            : Equation("Inviscid Burgers Equation", 1), diff_coef_(diff_coef) {
            if (_is_viscous()) {
                name_ = "Viscous Burgers Equation";
            }
        }

        double get_max_wave_speed(const torch::Tensor& u) override;

        double compute_cfl_dt(const torch::Tensor& u, double dx, double cfl_number) override;

        torch::Tensor compute_numerical_flux(
            const torch::Tensor& u, 
            const Mesh1d& mesh,
            const Reconstruction& reconstruction,
            const BoundaryCondition& boundary_condition,
            int n_ghost_cells) override;
                
        std::tuple<double, double> get_eigenvalues(const torch::Tensor& u);
};


class WaveEquation1d : public Equation {
    private:
        double wave_speed_;

    public:
        WaveEquation1d(double wave_speed)
            : Equation("Wave Equation", 2), wave_speed_(wave_speed) {}

        double get_max_wave_speed(const torch::Tensor& u) override;

        torch::Tensor compute_numerical_flux(
            const torch::Tensor& u, 
            const Mesh1d& mesh,
            const Reconstruction& reconstruction,
            const BoundaryCondition& boundary_condition,
            int n_ghost_cells) override;
                
        std::tuple<double, double> get_eigenvalues(const torch::Tensor& u);
};