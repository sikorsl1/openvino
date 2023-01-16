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

OutputVector generate_indices_from_repeats_tensor(std::vector<int64_t> repeats, NodeContext& context) {
    OutputVector all_indices;
    for (int i = 0; i < repeats.size(); i++) {
        Shape indices_shape{(unsigned int)repeats.at(i)};
        std::vector<int64_t> indices_vec(repeats.at(i), i);
        auto indices = context.mark_node(opset8::Constant::create(element::i64, indices_shape, indices_vec));
        all_indices.push_back(indices);
    }
    return all_indices;
};

OutputVector translate_repeat_interleave(NodeContext& context) {
    // constants
    auto const_0 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
    auto const_1 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {1}));
    auto const_neg_1 = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {-1}));

    // inputs
    auto input = context.get_input(0);
    auto repeats = context.const_input<std::vector<int64_t>>(1);

    auto const_repeats =
        context.mark_node(opset8::Constant::create(element::i64, Shape{2}, {repeats.at(0), (int64_t)1}));
    std::shared_ptr<ov::Node> result;

    if (context.input_is_none(2)) {
        if (repeats.size() == 1) {
            // case (repeats=number, dim=None)
            auto flat_shape = context.mark_node(opset8::Constant::create(element::i64, Shape{2}, {1, -1}));
            auto reshape = std::make_shared<opset8::Reshape>(input, flat_shape, false);
            auto tile = std::make_shared<opset8::Tile>(reshape, const_repeats);
            auto shape_perm = context.mark_node(opset8::Constant::create(element::i64, Shape{2}, {1, 0}));
            auto transpose = std::make_shared<opset8::Transpose>(tile, shape_perm);
            result = std::make_shared<opset8::Reshape>(transpose, const_neg_1, false);
        } else {
            // case (repeats=tensor, dim=None)
            auto flat_shape = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {-1}));
            auto reshape = std::make_shared<opset8::Reshape>(input, flat_shape, false);
            OutputVector all_indices = generate_indices_from_repeats_tensor(repeats, context);
            auto concat = std::make_shared<opset8::Concat>(all_indices, 0);
            result = std::make_shared<opset8::Gather>(reshape, concat, const_0);
        }
    } else {
        auto dim = context.const_input<int64_t>(2);
        auto const_dim = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {dim}));
        if (repeats.size() == 1) {
            // case (repeats=number, dim=number)
            auto input_shape = std::make_shared<opset8::ShapeOf>(input, element::i64);
            auto input_dim_size = std::make_shared<opset8::Gather>(input_shape, const_dim, const_0);
            auto range = std::make_shared<opset8::Range>(const_0, input_dim_size, const_1, element::i64);
            auto range_unsqeezed = std::make_shared<opset8::Unsqueeze>(range, const_0);
            auto tile = std::make_shared<opset8::Tile>(range_unsqeezed, const_repeats);
            auto shape_perm = context.mark_node(opset8::Constant::create(element::i64, Shape{2}, {1, 0}));
            auto transpose = std::make_shared<opset8::Transpose>(tile, shape_perm);
            auto flatten = std::make_shared<opset8::Reshape>(transpose, const_neg_1, false);
            result = std::make_shared<opset8::Gather>(input, flatten, const_dim);
        } else {
            // case (repeats=tensor, dim=number)
            OutputVector all_indices = generate_indices_from_repeats_tensor(repeats, context);
            auto concat = std::make_shared<opset8::Concat>(all_indices, 0);
            result = std::make_shared<opset8::Gather>(input, concat, const_dim);
        }
    }

    return {context.mark_node(result)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
