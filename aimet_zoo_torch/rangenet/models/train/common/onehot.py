# pylint: skip-file

# The MIT License
#
# Copyright (c) 2019 Andres Milioto, Jens Behley, Cyrill Stachniss, Photogrammetry and Robotics Lab, University of Bonn.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.functional as F
import __init__ as booger


class oneHot(nn.Module):
  def __init__(self, device, nclasses, spatial_dim=2):
    super().__init__()
    self.device = device
    self.nclasses = nclasses
    self.spatial_dim = spatial_dim

  def onehot1dspatial(self, x):
    # we only do tensors that 1d tensors that are batched or not, so check
    assert(len(x.shape) == 1 or len(x.shape) == 2)
    # if not batched, batch
    remove_dim = False  # flag to unbatch
    if len(x.shape) == 1:
      # add batch dimension
      x = x[None, ...]
      remove_dim = True

    # get tensor shape
    n, b = x.shape

    # scatter to onehot
    one_hot = torch.zeros((n, self.nclasses, b),
                          device=self.device).scatter_(1, x.unsqueeze(1), 1)

    # x is now [n,classes,b]

    # if it used to be unbatched, then unbatch it
    if remove_dim:
      one_hot = one_hot[0]

    return one_hot

  def onehot2dspatial(self, x):
    # we only do tensors that 2d tensors that are batched or not, so check
    assert(len(x.shape) == 2 or len(x.shape) == 3)
    # if not batched, batch
    remove_dim = False  # flag to unbatch
    if len(x.shape) == 2:
      # add batch dimension
      x = x[None, ...]
      remove_dim = True

    # get tensor shape
    n, h, w = x.shape

    # scatter to onehot
    one_hot = torch.zeros((n, self.nclasses, h, w),
                          device=self.device).scatter_(1, x.unsqueeze(1), 1)

    # x is now [n,classes,b]

    # if it used to be unbatched, then unbatch it
    if remove_dim:
      one_hot = one_hot[0]

    return one_hot

  def forward(self, x):
    # do onehot here
    if self.spatial_dim == 1:
      return self.onehot1dspatial(x)
    elif self.spatial_dim == 2:
      return self.onehot2dspatial(x)


if __name__ == "__main__":
  # get device
  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')

  # define number of classes
  nclasses = 6
  print("*"*80)
  print("Num classes 1d =", nclasses)
  print("*"*80)

  # test 1d unbatched case
  print("Tensor 1d spat dim, unbatched")
  tensor = torch.arange(0, nclasses).to(device)  # [0,1,2,3,4,5]
  print("in:", tensor)
  module = oneHot(device, nclasses, spatial_dim=1)
  print("out:", module(tensor))
  print("*"*80)

  # test 1d batched case
  print("*"*80)
  print("Tensor 1d spat dim, batched")
  tensor = torch.arange(0, nclasses).to(device)  # [0,1,2,3,4,5]
  tensor = torch.cat([tensor.unsqueeze(0),
                      tensor.unsqueeze(0)])      # [[0,1,2,3,4,5], [0,1,2,3,4,5]]
  print("in:", tensor)
  module = oneHot(device, nclasses, spatial_dim=1)
  print("out:", module(tensor))
  print("*"*80)

  # for 2 use less classes
  nclasses = 3
  print("*"*80)
  print("Num classes 2d =", nclasses)
  print("*"*80)

  # test 2d unbatched case
  print("*"*80)
  print("Tensor 2d spat dim, unbatched")
  tensor = torch.arange(0, nclasses).to(device)  # [0,1,2]
  tensor = torch.cat([tensor.unsqueeze(0),   # [[0,1,2],
                      tensor.unsqueeze(0),   # [0,1,2],
                      tensor.unsqueeze(0),   # [0,1,2],
                      tensor.unsqueeze(0)])  # [0,1,2]]
  print("in:", tensor)
  module = oneHot(device, nclasses, spatial_dim=2)
  print("out:", module(tensor))
  print("*"*80)

  # test 2d batched case
  print("*"*80)
  print("Tensor 2d spat dim, unbatched")
  tensor = torch.arange(0, nclasses).to(device)  # [0,1,2]
  tensor = torch.cat([tensor.unsqueeze(0),   # [[0,1,2],
                      tensor.unsqueeze(0),   # [0,1,2],
                      tensor.unsqueeze(0),   # [0,1,2],
                      tensor.unsqueeze(0)])  # [0,1,2]]
  tensor = torch.cat([tensor.unsqueeze(0),
                      tensor.unsqueeze(0)])  # 2 of the same 2d tensor
  print("in:", tensor)
  module = oneHot(device, nclasses, spatial_dim=2)
  print("out:", module(tensor))
  print("*"*80)
