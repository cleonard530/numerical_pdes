#include <iostream>
#include <string>
#include <torch/torch.h>

#include "mesh.h"
#include "boundary_conditions.h"
#include "reconstruction.h"
#include "equations.h"

constexpr double pi = 3.14159265358979323846;



int main() {
    double lb = -pi;
    double rb = pi;
    int n_cells = 100;

    std::cout << rb - lb << std::endl;

    Mesh1d mesh = Mesh1d(lb, rb, n_cells);

    std::cout << mesh.GetNCells() << std::endl;

    return 0;
}