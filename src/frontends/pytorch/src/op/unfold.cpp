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

OutputVector translate_unfold(NodeContext& context) {
    // constants
    auto const_0 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
    auto const_1 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {1}));
    auto const_0_list = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {0}));
    auto const_1_list = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {1}));

    // inputs
    auto input = context.get_input(0);
    int64_t dimension_int = context.const_input<int64_t>(1);
    auto dimension = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {dimension_int}));
    auto size = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {context.const_input<int64_t>(2)}));
    auto step = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {context.const_input<int64_t>(3)}));

    auto sizes = context.mark_node(std::make_shared<opset8::ShapeOf>(input));
    auto dimension_plus_1 = context.mark_node(std::make_shared<opset8::Add>(dimension, const_1_list));
    auto sizedim_tmp =
        context.mark_node(std::make_shared<opset8::Slice>(sizes, dimension, dimension_plus_1, const_1_list));
    auto sizedim = context.mark_node(std::make_shared<opset8::Reshape>(sizedim_tmp, const_1, false));
    auto sizedim_plus_1 = context.mark_node(std::make_shared<opset8::Add>(sizedim, const_1));

    auto low_indices = context.mark_node(std::make_shared<opset8::Range>(const_0, sizedim, step, element::i64));
    auto hi_indices = context.mark_node(std::make_shared<opset8::Range>(size, sizedim_plus_1, step, element::i64));

    auto ndim_tmp = context.mark_node(std::make_shared<opset8::ShapeOf>(sizes));
    auto ndim = context.mark_node(std::make_shared<opset8::Reshape>(ndim_tmp, const_1, false));
    auto dimension_scalar = context.mark_node(std::make_shared<opset8::Reshape>(dimension, const_1, false));
    auto dimension_plus_1_scalar =
        context.mark_node(std::make_shared<opset8::Reshape>(dimension_plus_1, const_1, false));
    auto perm_begin =
        context.mark_node(std::make_shared<opset8::Range>(const_0, dimension_scalar, const_1, element::i64));
    auto perm_end =
        context.mark_node(std::make_shared<opset8::Range>(dimension_plus_1_scalar, ndim, const_1, element::i64));
    auto perm = context.mark_node(std::make_shared<opset8::Concat>(OutputVector{perm_begin, perm_end, dimension}, 0));

    // body parameters
    auto input_param = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic());
    auto low_ind_param = std::make_shared<opset8::Parameter>(element::i64, PartialShape::dynamic());
    auto hi_ind_param = std::make_shared<opset8::Parameter>(element::i64, PartialShape::dynamic());
    auto perm_param = std::make_shared<opset8::Parameter>(element::i64, PartialShape::dynamic());
    auto iter_param = std::make_shared<opset8::Parameter>(element::i64, PartialShape::dynamic());

    // body
    auto iter_plus_1 = context.mark_node(std::make_shared<opset8::Add>(iter_param, const_1_list));
    auto low_ind_curr_iter = context.mark_node(
        std::make_shared<opset8::Slice>(low_ind_param, iter_param, iter_plus_1, const_1_list, const_0_list));
    auto hi_ind_curr_iter = context.mark_node(
        std::make_shared<opset8::Slice>(hi_ind_param, iter_param, iter_plus_1, const_1_list, const_0_list));
    auto slice = context.mark_node(
        std::make_shared<opset8::Slice>(input_param, low_ind_curr_iter, hi_ind_curr_iter, const_1_list, dimension));
    auto transpose = context.mark_node(std::make_shared<opset8::Transpose>(slice, perm_param));
    auto unsqueeze = context.mark_node(std::make_shared<opset8::Unsqueeze>(transpose, dimension));
    auto body =
        std::make_shared<Model>(OutputVector{unsqueeze},
                                ParameterVector{iter_param, input_param, low_ind_param, hi_ind_param, perm_param});

    // number of iterations
    auto low_indices_count = context.mark_node(std::make_shared<opset8::ShapeOf>(low_indices));
    auto hi_indices_count = context.mark_node(std::make_shared<opset8::ShapeOf>(hi_indices));
    auto iterations_count = context.mark_node(std::make_shared<opset8::Minimum>(low_indices_count, hi_indices_count));
    auto iterations_count_scalar =
        context.mark_node(std::make_shared<opset8::Reshape>(iterations_count, const_1, false));
    auto iter_values =
        context.mark_node(std::make_shared<opset8::Range>(const_0, iterations_count_scalar, const_1, element::i64));
    auto tensor_iterator = std::make_shared<opset8::TensorIterator>();

    // body input preparation
    tensor_iterator->set_function(body);
    tensor_iterator->set_invariant_input(input_param, input);
    tensor_iterator->set_invariant_input(perm_param, perm);
    tensor_iterator->set_invariant_input(low_ind_param, low_indices);
    tensor_iterator->set_invariant_input(hi_ind_param, hi_indices);
    tensor_iterator->set_sliced_input(iter_param, iter_values, 0, 1, 1, -1, 0);

    context.mark_nodes({tensor_iterator, input_param, low_ind_param, hi_ind_param, perm_param, iter_param});

    auto result = tensor_iterator->get_concatenated_slices(unsqueeze, 0, 1, 1, -1, dimension_int);
    return {context.mark_node(result.get_node_shared_ptr())};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov