# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestDtype(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array([1, 1]),)

    def create_model(self, dtype):

        class prim_dtype(torch.nn.Module):
            def __init__(self, dtype):
                super(prim_dtype, self).__init__()
                self.dtype = dtype

            def forward(self, input_tensor):
                input_tensor = input_tensor.to(self.dtype)
                return input_tensor.dtype

        ref_net = None

        return prim_dtype(dtype), ref_net, "prim::dtype"

    @pytest.mark.parametrize(("dtype"), [
        # torch.dtype, # corresponding int constant value
        torch.uint8, # 0
        torch.int8, # 1
        torch.int16, # 2
        torch.int32, # 3
        torch.int64, # 4
        torch.float16, # 5
        torch.float32, # 6
        torch.float64, # 7
        torch.bool, # 11
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_dtype(self, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(dtype), ie_device, precision, ir_version)
