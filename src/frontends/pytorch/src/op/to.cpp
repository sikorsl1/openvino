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

OutputVector translate_to(NodeContext& context) {
    int dtype_idx;
    int non_blocking_idx;
    int copy_idx;
    int memory_format_idx;
    if (context.get_input_size() == 5) {
        // aten::to.dtype(Tensor(a) self, int dtype, bool non_blocking=False, bool copy=False, int? memory_format=None)
        // -> (Tensor(a))
        dtype_idx = 1;
        non_blocking_idx = 2;
        copy_idx = 3;
        memory_format_idx = 4;
    } else if (context.get_input_size() == 6) {
        // aten::to.device(Tensor(a) self, Device device, int dtype, bool non_blocking=False, bool copy=False, int?
        // memory_format=None) -> (Tensor(a)).
        // Input with index 1 is device we skip that input.
        dtype_idx = 2;
        non_blocking_idx = 3;
        copy_idx = 4;
        memory_format_idx = 5;
    } else {
        FRONT_END_OP_CONVERSION_CHECK(false, "Unknown aten::to format");
    }
    FRONT_END_OP_CONVERSION_CHECK(
        context.input_is_none(non_blocking_idx) || context.const_input<bool>(non_blocking_idx) == false,
        "aten::to translation do not support non_blocking attribute");
    FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(copy_idx) || context.const_input<bool>(copy_idx) == false,
                                  "aten::to translation do not support copy attribute");
    FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(memory_format_idx),
                                  "aten::to translation do not support memory_format attribute");
    auto dtype_ext_node = context.get_input_from_visible_context(dtype_idx).get_node_shared_ptr();
    auto dtype_fw_node = std::dynamic_pointer_cast<PtFrameworkNode>(dtype_ext_node);
    Output<Node> cast;
    if (dtype_fw_node && dtype_fw_node->get_op_type() == "prim::dtype") {
        auto type_input = dtype_fw_node->input_value(0);
        cast = context.mark_node(std::make_shared<opset8::ConvertLike>(context.get_input(0), type_input));
    } else if (const auto dtype_const = std::dynamic_pointer_cast<opset8::Constant>(dtype_ext_node)) {
        auto pt_type = dtype_const->cast_vector<int64_t>()[0];
        FRONT_END_OP_CONVERSION_CHECK(TORCH_TO_OV_TYPE.count(pt_type), "Unknown type in aten::to: ", pt_type);
        auto dtype = TORCH_TO_OV_TYPE.at(pt_type);
        cast = context.mark_node(std::make_shared<opset8::Convert>(context.get_input(0), dtype));
    } else {
        cast = context.mark_node(std::make_shared<opset8::ConvertLike>(context.get_input(0), context.get_input(1)));
    }
    return {cast};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov