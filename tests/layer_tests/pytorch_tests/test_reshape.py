# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest


class TestReshape(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.uniform(0, 50, 6).astype(np.float32),)

    def create_model(self, shape):

        import torch

        class aten_reshape(torch.nn.Module):
            def __init__(self, shape):
                super(aten_reshape, self).__init__()
                self.shape = shape


            def forward(self, x):
                return torch.reshape(x, self.shape)

        ref_net = None

        return aten_reshape(shape), ref_net, "aten::reshape"

    @pytest.mark.parametrize(("shape"), [
        [2, 3],
        [3, 2],
        [3, 2],
        [6, 1],
        [1, 6],
    ])
    @pytest.mark.nightly
    def test_reshape(self, shape, ie_device, precision, ir_version):
    	if ie_device == "CPU":
            self._test(*self.create_model(shape), ie_device, precision, ir_version)
