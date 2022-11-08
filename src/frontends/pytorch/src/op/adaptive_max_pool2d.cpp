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

OutputVector translate_adaptive_max_pool2d(NodeContext& context) {
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    auto adaptive_max_pool = context.mark_node(std::make_shared<opset8::AdaptiveMaxPool>(x, y));
    auto return_indices = context.const_input<bool>(2);
    OutputVector res{adaptive_max_pool->output(0)};
    if (return_indices) {
        res.push_back(adaptive_max_pool->output(1));
    }
    return res;
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov