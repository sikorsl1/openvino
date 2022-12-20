# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestRSqrt(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 10).astype(np.float32),)

    def create_model(self):

        import torch
        import torch.nn.functional as F

        class aten_rsqrt(torch.nn.Module):

            def forward(self, x):
                return torch.rsqrt(x)

        ref_net = None

        return aten_rsqrt(), ref_net, "aten::rsqrt"

    @pytest.mark.nightly
    def test_relu(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)