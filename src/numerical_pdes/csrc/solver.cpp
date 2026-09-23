#include <iostream>
#include <string>
#include <torch/torch.h>

#include "mesh.h"
#include "boundary_conditions.h"
#include "reconstruction.h"
#include "equations.h"


torch::Tensor solve1d(
    Equation equation,
    Mesh1d mesh,
    BoundaryCondition boundary_condition,
    TimeIntegrator time_integrator,
    Reconstruction reconstruction,
    double cfl,
    const torch::Tensor& u0,
    const torch::Tensor& t_span) {

        
}

