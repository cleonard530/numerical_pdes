#include <gtest/gtest.h>
#include "mesh.h"
#include "test_utils.h"
#include <iostream>

TEST(Mesh1dTest, TestMeshMethods) {
  Mesh1d mesh(-1, 1, 10);

  EXPECT_EQ(mesh.GetNDim(), 1);
  EXPECT_EQ(mesh.GetNCells(), 10);
    
  mesh.SetNCells(20);
  EXPECT_EQ(mesh.GetNCells(), 20);

}


TEST(Mesh1dTest, TestGetters) {
  Mesh1d mesh(-1, 1, 10);

  EXPECT_NEAR(mesh.GetDx(), 2.0 / 10.0, 1e-8);
  
  torch::Tensor expected_cell_centers = 
    torch::tensor({-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9}, torch::kFloat64);  
  bool are_close = test_utils::AreTensorClose(mesh.GetCellCenters(), expected_cell_centers);  
  EXPECT_TRUE(are_close);
}


TEST(Mesh1dTest, TestGetCellCentersWithGhostCells) {
  Mesh1d mesh(-1, 1, 10);

  torch::Tensor expected_cell_centers_with_gc1 = 
    torch::tensor({-1.1, -0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1}, torch::kFloat64);
  torch::Tensor ccwg1 = mesh.GetCellCentersWithGhostCells(1);
  bool are_close_with_gc1 = 
    test_utils::AreTensorClose(ccwg1, expected_cell_centers_with_gc1);  
  EXPECT_TRUE(are_close_with_gc1);
}


TEST(Mesh1dTest, TestGetPeriod) {
  Mesh1d mesh(-1, 1, 10);

  EXPECT_NEAR(mesh.GetPeriod(), 2.0, 1e-12);
}