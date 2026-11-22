#pragma once
#include <torch/torch.h>


namespace test_utils {

bool AreTensorClose(torch::Tensor t1, torch::Tensor t2, double rtol=1e-5) {
    return torch::allclose(t1, t2, rtol);
}

}





