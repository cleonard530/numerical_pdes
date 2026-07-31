#include <torch/torch.h>


class Mesh {
    private:
        int n_dim;

        int n_cells;

    public:
        Mesh(int n_dim_in, int n_cells_in)
            : n_dim(n_dim_in), n_cells(n_cells_in) {}

        int GetNDim() const { return n_dim; }
        void SetNDim(int n_dim_in) { n_dim = n_dim_in; }

        int GetNCells() const { return n_cells; }
        void SetNCells(int n_cells_in) { n_cells = n_cells_in; }
};


class Mesh1d : public Mesh {
    private:
        double x_left;
        double x_right;
        double dx;
        torch::Tensor cell_centers;


    public:
        Mesh1d(double x_left_in, double x_right_in, int n_cells_in)
            : Mesh(1, n_cells_in),
              x_left(x_left_in),
              x_right(x_right_in),
              dx((x_right_in-x_left_in) / n_cells_in),
              cell_centers(torch::linspace(
                x_left_in + dx / 2.0,
                x_right_in - dx / 2.0,
                n_cells_in,
                torch::kFloat64)) {}


        double Period();
};