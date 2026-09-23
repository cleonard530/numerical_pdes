#pragma once
#include <torch/torch.h>


class Mesh {
    private:
        int n_dim_;

        int n_cells_;

    public:
        Mesh(int n_dim, int n_cells)
            : n_dim_(n_dim), n_cells_(n_cells) {}

        int GetNDim() const { return n_dim_; }
        void SetNDim(int n_dim) { n_dim_ = n_dim; }

        int GetNCells() const { return n_cells_; }
        void SetNCells(int n_cells) { n_cells_ = n_cells; }
};


class Mesh1d : public Mesh {
    private:
        double x_left_;
        double x_right_;
        double dx_;
        torch::Tensor cell_centers_;

    public:
        Mesh1d(double x_left, double x_right, int n_cells)
            : Mesh(1, n_cells),
              x_left_(x_left),
              x_right_(x_right),
              dx_((x_right-x_left) / n_cells),
              cell_centers_(torch::linspace(
                x_left + dx_ / 2.0,
                x_right - dx_ / 2.0,
                n_cells,
                torch::kFloat64)) {}
        
        double GetDx() const { return dx_; }
        void SetDx(double dx) { dx_ = dx; }

        torch::Tensor GetCellCenters() const { return cell_centers_; }
        void SetCellCenters(torch::Tensor cell_centers) { cell_centers_ = cell_centers; }

        double GetPeriod();
};