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

OutputVector translate_int(NodeContext& context) {
    return {context.mark_node(std::make_shared<opset8::Convert>(context.get_input(0), element::i64))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov