// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_conv2d(NodeContext& context) {
    auto strides = context.const_input<Strides>(3);
    // In torch pads at beginning are same as at end
    auto pads = CoordinateDiff(strides.size(), 0);
    auto pad_type = ov::op::PadType::EXPLICIT;
    try {
        auto pad_mode = context.const_input<std::string>(4);
        static std::unordered_multimap<std::string, ov::op::PadType> auto_pad_values{
            {"valid", ov::op::PadType::VALID},
            {"same", ov::op::PadType::SAME_UPPER}};

        const auto auto_pad_type_ptr = auto_pad_values.find(pad_mode);
        FRONT_END_OP_CONVERSION_CHECK(auto_pad_type_ptr != auto_pad_values.end(),
                                      "Provided `padding` value: '",
                                      pad_mode,
                                      "' is invalid.");
        pad_type = auto_pad_type_ptr->second;
    } catch (ov::frontend::GeneralFailure) {
        pads = context.const_input<CoordinateDiff>(4);
    }
    auto dilations = context.const_input<Strides>(5);
    auto groups = context.const_input<int64_t>(6);

    std::shared_ptr<ov::Node> conv;
    if (groups == 1) {
        conv = std::make_shared<opset8::Convolution>(context.get_input(0),
                                                     context.get_input(1),
                                                     strides,
                                                     pads,
                                                     pads,
                                                     dilations,
                                                     pad_type);
    } else {
        conv = std::make_shared<opset8::GroupConvolution>(
            context.get_input(0),
            reshape_kernel_for_group(context, context.get_input(0), context.get_input(1), groups),
            strides,
            pads,
            pads,
            dilations,
            pad_type);
    }

    return {context.mark_output(make_optional_bias(conv, context, 2, {-2, -1}))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov