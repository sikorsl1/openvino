// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_dtype(NodeContext& context) {
    auto input = context.get_input(0);
    element::Type ov_type = input.get_element_type();
    FRONT_END_OP_CONVERSION_CHECK(ov_type != element::dynamic && ov_type != element::undefined,
                                  "prim::dtype conversion supports only static data types.");

    std::map<element::Type, int> ov_to_pt_type{{element::u8, 0},
                                               {element::i8, 1},
                                               {element::i16, 2},
                                               {element::i32, 3},
                                               {element::i64, 4},
                                               {element::f16, 5},
                                               {element::f32, 6},
                                               {element::f64, 7},
                                               {element::boolean, 11}};

    auto pt_type = ov_to_pt_type.find(ov_type);
    FRONT_END_OP_CONVERSION_CHECK(pt_type != ov_to_pt_type.end(),
                                  "prim::dtype conversion doesn't support [ ",
                                  ov_type,
                                  " ] data type.");
    auto type_const = context.mark_node(v0::Constant::create(element::i32, Shape{}, {pt_type->second}));
    return {type_const};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
