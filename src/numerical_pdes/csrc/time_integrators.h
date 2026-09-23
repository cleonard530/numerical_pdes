#pragma once
#include <torch/torch.h>
#include <string>
#include <tuple>

#include "mesh.h"
#include "boundary_conditions.h"
#include "reconstruction.h"


class TimeIntegrator {
  public:
    virtual void step(
      torch::Tensor& u,
      double dt,
      BoundaryCondition boundary_condition,
      int n_ghost_cells) const = 0;

    virtual torch::Tensor compute_rhs_1d(
      Equations equation, 
      Mesh1d mesh, 
      Reconstruction reconstruction,
      BoundaryCondition boundary_condition,
      int n_ghost_cells) const;
};


class ForwardEuler : TimeIntegrator {
  public:
    void step(
      torch::Tensor& u,
      double dt,
      BoundaryCondition boundary_condition,
      int n_ghost_cells) const override;
};


class SSPRK3 : TimeIntegrator {
  public:
    void step(
      torch::Tensor& u,
      double dt,
      BoundaryCondition boundary_condition,
      int n_ghost_cells) const override;
};