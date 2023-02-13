// BSD 3-Clause License
//
// Copyright (c) 2017, mapillary
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#include <torch/torch.h>

#include <vector>

#include "inplace_abn.h"

std::vector<at::Tensor> mean_var(at::Tensor x) {
  if (x.is_cuda()) {
    return mean_var_cuda(x);
  } else {
    return mean_var_cpu(x);
  }
}

at::Tensor forward(at::Tensor x, at::Tensor mean, at::Tensor var, at::Tensor weight, at::Tensor bias,
                   bool affine, float eps) {
  if (x.is_cuda()) {
    return forward_cuda(x, mean, var, weight, bias, affine, eps);
  } else {
    return forward_cpu(x, mean, var, weight, bias, affine, eps);
  }
}

std::vector<at::Tensor> edz_eydz(at::Tensor z, at::Tensor dz, at::Tensor weight, at::Tensor bias,
                                 bool affine, float eps) {
  if (z.is_cuda()) {
    return edz_eydz_cuda(z, dz, weight, bias, affine, eps);
  } else {
    return edz_eydz_cpu(z, dz, weight, bias, affine, eps);
  }
}

std::vector<at::Tensor> backward(at::Tensor z, at::Tensor dz, at::Tensor var, at::Tensor weight, at::Tensor bias,
                                 at::Tensor edz, at::Tensor eydz, bool affine, float eps) {
  if (z.is_cuda()) {
    return backward_cuda(z, dz, var, weight, bias, edz, eydz, affine, eps);
  } else {
    return backward_cpu(z, dz, var, weight, bias, edz, eydz, affine, eps);
  }
}

void leaky_relu_forward(at::Tensor z, float slope) {
  at::leaky_relu_(z, slope);
}

void leaky_relu_backward(at::Tensor z, at::Tensor dz, float slope) {
  if (z.is_cuda()) {
    return leaky_relu_backward_cuda(z, dz, slope);
  } else {
    return leaky_relu_backward_cpu(z, dz, slope);
  }
}

void elu_forward(at::Tensor z) {
  at::elu_(z);
}

void elu_backward(at::Tensor z, at::Tensor dz) {
  if (z.is_cuda()) {
    return elu_backward_cuda(z, dz);
  } else {
    return elu_backward_cpu(z, dz);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mean_var", &mean_var, "Mean and variance computation");
  m.def("forward", &forward, "In-place forward computation");
  m.def("edz_eydz", &edz_eydz, "First part of backward computation");
  m.def("backward", &backward, "Second part of backward computation");
  m.def("leaky_relu_forward", &leaky_relu_forward, "Leaky relu forward computation");
  m.def("leaky_relu_backward", &leaky_relu_backward, "Leaky relu backward computation and inversion");
  m.def("elu_forward", &elu_forward, "Elu forward computation");
  m.def("elu_backward", &elu_backward, "Elu backward computation and inversion");
}