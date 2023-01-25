# =================================================================================
#
# MIT License
#
# Copyright (c) 2019 Hao Gao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# =================================================================================

from collections import OrderedDict
import torch.nn as nn


class ModelBook:
    """Maintain the mapping between modules and their paths.

    Example:
        book = ModelBook(model_ft)
        for p, m in book.conv2d_modules():
            print('path:', p, 'num of filters:', m.out_channels)
            assert m is book.get_module(p)
    """

    def __init__(self, model):
        self._model = model
        self._modules = OrderedDict()
        self._paths = OrderedDict()
        path = []
        self._construct(self._model, path)

    def _construct(self, module, path):
        if not module._modules:
            return
        for name, m in module._modules.items():
            cur_path = tuple(path + [name])
            self._paths[m] = cur_path
            self._modules[cur_path] = m
            self._construct(m, path + [name])

    def conv2d_modules(self):
        return self.modules(nn.Conv2d)

    def linear_modules(self):
        return self.modules(nn.Linear)

    def modules(self, module_type=None):
        for p, m in self._modules.items():
            if not module_type or isinstance(m, module_type):
                yield p, m

    def num_of_conv2d_modules(self):
        return self.num_of_modules(nn.Conv2d)

    def num_of_conv2d_filters(self):
        """Return the sum of out_channels of all conv2d layers.

        Here we treat the sub weight with size of [in_channels, h, w] as a single filter.
        """
        num_filters = 0
        for _, m in self.conv2d_modules():
            num_filters += m.out_channels
        return num_filters

    def num_of_linear_modules(self):
        return self.num_of_modules(nn.Linear)

    def num_of_linear_filters(self):
        num_filters = 0
        for _, m in self.linear_modules():
            num_filters += m.out_features
        return num_filters

    def num_of_modules(self, module_type=None):
        num = 0
        for p, m in self._modules.items():
            if not module_type or isinstance(m, module_type):
                num += 1
        return num

    def get_module(self, path):
        return self._modules.get(path)

    def get_path(self, module):
        return self._paths.get(module)

    def update(self, path, module):
        old_module = self._modules[path]
        del self._paths[old_module]
        self._paths[module] = path
        self._modules[path] = module
