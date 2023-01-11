// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_batch_norm(NodeContext& context) {
    // Schema: aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var,
    // bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor
    auto input = context.get_input(0);
    auto input_shape = context.mark_node(std::make_shared<opset8::ShapeOf>(input));
    Output<Node> weight;
    Output<Node> bias;
    if (!context.input_is_none(1)){ 
        weight = context.get_input(1);
    }
    else {
        auto zero_i = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
        auto one_i = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {1}));
        auto one_f = context.mark_node(opset8::Constant::create(element::f32, Shape{}, {1}));
        auto channel_dim = context.mark_node(std::make_shared<opset8::Gather>(input_shape, one_i, zero_i));
        auto channel_dim_exp = context.mark_node(std::make_shared<opset8::Unsqueeze>(channel_dim, zero_i));
        weight = context.mark_node(std::make_shared<opset8::Broadcast>(one_f, channel_dim_exp));
    }
    if (!context.input_is_none(2)){
        bias = context.get_input(2);
    }
    else {
        auto zero_i = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
        auto one_i = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {1}));
        auto channel_dim = context.mark_node(std::make_shared<opset8::Gather>(input_shape, one_i, zero_i));
        auto channel_dim_exp = context.mark_node(std::make_shared<opset8::Unsqueeze>(channel_dim, zero_i));
        auto zero_f = context.mark_node(opset8::Constant::create(element::f32, Shape{}, {0}));
        bias = context.mark_node(std::make_shared<opset8::Broadcast>(zero_f, channel_dim_exp));
    }
    // index 3 running_mean and index 4 running_var can be none for training case only, check that not training before
    auto training = context.const_input<bool>(5);
    FRONT_END_OP_CONVERSION_CHECK(!training, "Translation for aten::batch_norm do not support training mode.");
    auto running_mean = context.get_input(3);
    auto running_var = context.get_input(4);
    // Index with index 6 is momentum, it is used only in training mode
    auto epsilon = context.const_input<float>(7);
    return {context.mark_node(
        std::make_shared<opset10::BatchNormInference>(input, weight, bias, running_mean, running_var, epsilon))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov