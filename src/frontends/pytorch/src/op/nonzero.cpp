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

OutputVector translate_nonzero(NodeContext& context) {
    auto cond = context.get_input(0);
    auto non_zero = context.mark_node(std::make_shared<opset8::NonZero>(cond));
    auto input_order = context.mark_node(opset8::Constant::create(element::i64, Shape{2}, {1, 0}));
    return {context.mark_node(std::make_shared<opset8::Transpose>(non_zero, input_order))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov