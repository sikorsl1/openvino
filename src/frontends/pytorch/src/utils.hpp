// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/opsets/opset8.hpp>

#include "openvino/frontend/pytorch/node_context.hpp"

namespace ov {

namespace op {
namespace util {
class FrameworkNode;
}
}  // namespace op

namespace frontend {
namespace pytorch {

const std::map<int, element::Type> TORCH_TO_OV_TYPE{{0, element::u8},
                                                    {1, element::i8},
                                                    {2, element::i16},
                                                    {3, element::i32},
                                                    {4, element::i64},
                                                    {5, element::f16},
                                                    {6, element::f32},
                                                    {7, element::f64},
                                                    {11, element::boolean}};

const std::unordered_multimap<std::string, ov::op::PadType> TORCH_AUTO_PAD_TO_OV{{"valid", ov::op::PadType::VALID},
                                                                                 {"same", ov::op::PadType::SAME_UPPER}};

Output<Node> make_optional_bias(const Output<Node>& base_op,
                                const NodeContext& context,
                                size_t bias_input_idx,
                                const std::vector<int>& unsqueeze_dims = {});

Output<ov::Node> reshape_conv_bias(NodeContext& context, Output<ov::Node> bias, Output<ngraph::Node> conv);

std::shared_ptr<ov::Node> get_rank_node(const Output<Node>& node);

Output<Node> reshape_kernel_for_group(const NodeContext& context,
                                      const Output<Node>& input,
                                      const Output<Node>& kernel,
                                      int64_t groups);

std::shared_ptr<Node> get_axes_range(NodeContext& context, size_t input_id);

std::shared_ptr<Node> numel(NodeContext& context, size_t input_id);

ov::element::Type convert_dtype(NodeContext& context, size_t input_id);

std::shared_ptr<Node> concat_list_construct(std::shared_ptr<Node> input);

std::shared_ptr<ov::Model> convert_pytorch_model(std::shared_ptr<Decoder> pytorch_model,
                                                 const TensorMap& external_tensor_map = {});

OutputVector convert_node(NodeContext* context);

std::shared_ptr<ov::op::util::FrameworkNode> cast_fw_node(std::shared_ptr<Node> node, const std::string& type);

// TODO: Elimitate the need of this function by implementing more accurate custom data type handling
Any simplified_type_interpret(Any type);

namespace op {
template <OutputVector (*T)(NodeContext&), size_t idx = 0>
OutputVector inplace_op(NodeContext& context) {
    auto translation_res = T(context);
    FRONT_END_OP_CONVERSION_CHECK(translation_res.size() == 1,
                                  "inplace_op function must be used on single output translators");
    context.mutate_input(idx, translation_res[0]);
    return translation_res;
}

template <typename T>
OutputVector translate_1to1_match_1_inputs(NodeContext& context) {
    auto inputs = context.inputs();
    FRONT_END_OP_CONVERSION_CHECK(inputs.size() >= 1, "Operation has no inputs.");
    for (int i = 1; i < inputs.size(); i++) {
        FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(i), "Got more inputs than expected.");
    }
    FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(0), "Input should not be None.");
    return {context.mark_node(std::make_shared<T>(inputs[0]))};
}

template <typename T>
OutputVector translate_1to1_match_2_inputs(NodeContext& context) {
    auto inputs = context.inputs();
    FRONT_END_OP_CONVERSION_CHECK(inputs.size() >= 2, "Operation has less then 2 inputs.");
    for (int i = 2; i < inputs.size(); i++) {
        FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(i), "Got more inputs than expected.");
    }
    FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(0) && !context.input_is_none(1), "Inputs should not be None.");
    return {context.mark_node(std::make_shared<T>(inputs[0], inputs[1]))};
}

inline OutputVector return_false_scalar(NodeContext& context) {
    return {context.mark_node(opset8::Constant::create(element::boolean, Shape{}, {false}))};
}

inline OutputVector skip_node(NodeContext& context) {
    return {context.get_input(0).get_node_shared_ptr()};
}
}  // namespace op

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
