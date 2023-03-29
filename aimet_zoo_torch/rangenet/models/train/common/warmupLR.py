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


import torch.optim.lr_scheduler as toptim


class warmupLR(toptim._LRScheduler):
  """ Warmup learning rate scheduler.
      Initially, increases the learning rate from 0 to the final value, in a
      certain number of steps. After this number of steps, each step decreases
      LR exponentially.
  """

  def __init__(self, optimizer, lr, warmup_steps, momentum, decay):
    # cyclic params
    self.optimizer = optimizer
    self.lr = lr
    self.warmup_steps = warmup_steps
    self.momentum = momentum
    self.decay = decay

    # cap to one
    if self.warmup_steps < 1:
      self.warmup_steps = 1

    # cyclic lr
    self.initial_scheduler = toptim.CyclicLR(self.optimizer,
                                             base_lr=0,
                                             max_lr=self.lr,
                                             step_size_up=self.warmup_steps,
                                             step_size_down=self.warmup_steps,
                                             cycle_momentum=False,
                                             base_momentum=self.momentum,
                                             max_momentum=self.momentum)

    # our params
    self.last_epoch = -1  # fix for pytorch 1.1 and below
    self.finished = False  # am i done
    super().__init__(optimizer)

  def get_lr(self):
    return [self.lr * (self.decay ** self.last_epoch) for lr in self.base_lrs]

  def step(self, epoch=None):
    if self.finished or self.initial_scheduler.last_epoch >= self.warmup_steps:
      if not self.finished:
        self.base_lrs = [self.lr for lr in self.base_lrs]
        self.finished = True
      return super(warmupLR, self).step(epoch)
    else:
      return self.initial_scheduler.step(epoch)
