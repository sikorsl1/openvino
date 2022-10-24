# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestRelu(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, dim):

        import torch
        import torch.nn.functional as F

        class aten_softmax(torch.nn.Module):
            def __init__(self, dim):
                super(aten_softmax, self).__init__()
                self.dim = dim

            def forward(self, x):
                return F.softmax(x, self.dim)

        ref_net = None

        return aten_softmax(dim), ref_net

    @pytest.mark.parametrize("dim", [-1, 3])
    @pytest.mark.nightly
    def test_relu(self, dim, ie_device, precision, ir_version):
        self._test(*self.create_model(dim), ie_device, precision, ir_version)
