# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestLeakyRelu(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, alpha, inplace):
        import torch
        import torch.nn.functional as F

        class aten_leaky_relu(torch.nn.Module):
            def __init__(self, alpha, inplace):
                super(aten_leaky_relu, self).__init__()
                self.alpha = alpha
                self.inplace = inplace

            def forward(self, x):
                return torch.cat([x, F.leaky_relu(x, self.alpha, inplace=self.inplace)])
            
        ref_net = None

        return aten_leaky_relu(alpha, inplace), ref_net, "aten::leaky_relu" if not inplace else "aten::leaky_relu_"


    @pytest.mark.parametrize("alpha,inplace", [(0.01, True), (0.01, False), (1.01, True), (1.01, False), (-0.01, True), (-0.01, False)])
    @pytest.mark.nightly
    def test_leaky_relu(self, alpha, inplace, ie_device, precision, ir_version):
        self._test(*self.create_model(alpha, inplace), ie_device, precision, ir_version)