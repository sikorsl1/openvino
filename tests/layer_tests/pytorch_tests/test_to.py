# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest
from openvino.pyopenvino import OpConversionFailure 


class TestAtenTo(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(1, 3).astype(self.input_type),)

    def create_model(self, type, non_blocking=False, copy=False, memory_format=None):

        import torch
        import torch.nn.functional as F

        class aten_to(torch.nn.Module):
            def __init__(self, type, non_blocking=False, copy=False, memory_format=None):
                super(aten_to, self).__init__()            
                self.type = type
                self.non_blocking = non_blocking
                self.copy = copy
                self.memory_format = memory_format

            def forward(self, x):
                return x.to(self.type, self.non_blocking, self.copy, self.memory_format)
                # return x.to(self.type)

        ref_net = None

        return aten_to(type, non_blocking, copy, memory_format), ref_net

    # Cartesian product of input/output types
    @pytest.mark.parametrize("input_type", [np.int32, np.float32, np.float64])
    @pytest.mark.parametrize("output_type", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.float32, torch.int64])
    @pytest.mark.nightly
    def test_aten_to(self, input_type, output_type, ie_device, precision, ir_version):
        if ie_device == "CPU":
            self.input_type = input_type
            self._test(*self.create_model(output_type), ie_device, precision, ir_version)

    # Cartesian product of input/output types
    @pytest.mark.parametrize("input_type", [np.int32, np.float32, np.float64])
    @pytest.mark.parametrize(("output_type", "non_blocking"), [
            [torch.uint8, True],
            [torch.int8, True],
            [torch.int16, True],
            [torch.int32, True],
            [torch.int64, True],
            [torch.float32, True],
            [torch.float64, True],
    ])
    @pytest.mark.nightly
    def test_aten_to_raise_non_blocking_arg(self, input_type, output_type, non_blocking, ie_device, precision, ir_version):
        if ie_device == "CPU":
            self.input_type = input_type
            with pytest.raises(OpConversionFailure) as e:
                self._test(*self.create_model(output_type, non_blocking=non_blocking), ie_device, precision, ir_version) 


    # Cartesian product of input/output types
    @pytest.mark.parametrize("input_type", [np.int32, np.float32, np.float64])
    @pytest.mark.parametrize(("output_type", "copy"), [
            [torch.uint8, True],
            [torch.int8, True],
            [torch.int16, True],
            [torch.int32, True],
            [torch.int64, True],
            [torch.float32, True],
            [torch.float64, True],
    ])
    @pytest.mark.nightly
    def test_aten_to_raise_copy_arg(self, input_type, output_type, copy, ie_device, precision, ir_version):
        if ie_device == "CPU":
            self.input_type = input_type
            with pytest.raises(OpConversionFailure) as e:
                self._test(*self.create_model(output_type, copy=copy), ie_device, precision, ir_version) 

    # Cartesian product of input/output types
    @pytest.mark.parametrize("input_type", [np.int32, np.float32, np.float64])
    @pytest.mark.parametrize(("output_type", "memory_format"), [
            [torch.uint8, 1],
            [torch.int8, 1],
            [torch.int16, 2],
            [torch.int32, 2],
            [torch.int64, 3],
            [torch.float32, 3],
            [torch.float64, 4],
    ])
    @pytest.mark.nightly
    def test_aten_to_raise_memory_format_arg(self, input_type, output_type, memory_format, ie_device, precision, ir_version):
        if ie_device == "CPU":
            self.input_type = input_type
            with pytest.raises(OpConversionFailure) as e:
                self._test(*self.create_model(output_type, memory_format=memory_format), ie_device, precision, ir_version) 
