// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"

#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

#define OP_CONVERTER(op) OutputVector op(NodeContext& node)

OP_CONVERTER(translate_adaptive_avg_pool3d);
OP_CONVERTER(translate_adaptive_max_pool2d);
OP_CONVERTER(translate_add);
OP_CONVERTER(translate_addcmul);
OP_CONVERTER(translate_addmm);
OP_CONVERTER(translate_as_tensor);
OP_CONVERTER(translate_avg_pool2d);
OP_CONVERTER(translate_batch_norm);
OP_CONVERTER(translate_clamp);
OP_CONVERTER(translate_constant);
OP_CONVERTER(translate_conv2d);
OP_CONVERTER(translate_convolution);
OP_CONVERTER(translate_convolution_mode);
OP_CONVERTER(translate_dim);
OP_CONVERTER(translate_div);
OP_CONVERTER(translate_elu);
OP_CONVERTER(translate_expand);
OP_CONVERTER(translate_expand_as);
OP_CONVERTER(translate_embedding);
OP_CONVERTER(translate_flatten);
OP_CONVERTER(translate_floordiv);
OP_CONVERTER(translate_full);
OP_CONVERTER(translate_full_like);
OP_CONVERTER(translate_gelu);
OP_CONVERTER(translate_group_norm);
OP_CONVERTER(translate_hardtanh);
OP_CONVERTER(translate_if);
OP_CONVERTER(translate_im2col);
OP_CONVERTER(translate_int);
OP_CONVERTER(translate_layer_norm);
OP_CONVERTER(translate_linear);
OP_CONVERTER(translate_loop);
OP_CONVERTER(translate_max_pool2d);
OP_CONVERTER(translate_max);
OP_CONVERTER(translate_masked_fill);
OP_CONVERTER(translate_mean);
OP_CONVERTER(translate_min);
OP_CONVERTER(translate_neg);
OP_CONVERTER(translate_norm);
OP_CONVERTER(translate_new_full);
OP_CONVERTER(translate_new_ones);
OP_CONVERTER(translate_new_zeros);
OP_CONVERTER(translate_numel);
OP_CONVERTER(translate_ones);
OP_CONVERTER(translate_ones_like);
OP_CONVERTER(translate_pad);
OP_CONVERTER(translate_reciprocal);
OP_CONVERTER(translate_relu6);
OP_CONVERTER(translate_reshape);
OP_CONVERTER(translate_reshape_as);
OP_CONVERTER(translate_rsub);
OP_CONVERTER(translate_roll);
OP_CONVERTER(translate_rsqrt);
OP_CONVERTER(translate_select);
OP_CONVERTER(translate_size);
OP_CONVERTER(translate_slice);
OP_CONVERTER(translate_softmax);
OP_CONVERTER(translate_square);
OP_CONVERTER(translate_squeeze);
OP_CONVERTER(translate_sub);
OP_CONVERTER(translate_sum);
OP_CONVERTER(translate_to);
OP_CONVERTER(translate_transpose);
OP_CONVERTER(translate_tuple_construct);
OP_CONVERTER(translate_unfold);
OP_CONVERTER(translate_upsample_bicubic2d);
OP_CONVERTER(translate_upsample_bilinear2d);
OP_CONVERTER(translate_upsample_nearest2d);
OP_CONVERTER(translate_var);
OP_CONVERTER(translate_view);
OP_CONVERTER(translate_zeros);
OP_CONVERTER(translate_zeros_like);

}  // namespace op

const std::map<std::string, CreatorFunction> get_supported_ops() {
    return {
        {"aten::_convolution", op::translate_convolution},
        {"aten::_convolution_mode", op::translate_convolution_mode},
        {"aten::abs", op::translate_1to1_match_1_inputs<opset8::Abs>},
        {"aten::acos", op::translate_1to1_match_1_inputs<opset8::Acos>},
        {"aten::acos_", op::inplace_op<op::translate_1to1_match_1_inputs<opset8::Acos>>},
        {"aten::acosh", op::translate_1to1_match_1_inputs<opset8::Acosh>},
        {"aten::acosh_", op::inplace_op<op::translate_1to1_match_1_inputs<opset8::Acosh>>},
        {"aten::adaptive_avg_pool2d", op::translate_1to1_match_2_inputs<opset8::AdaptiveAvgPool>},
        {"aten::adaptive_avg_pool3d", op::translate_adaptive_avg_pool3d},
        {"aten::adaptive_max_pool2d", op::translate_adaptive_max_pool2d},
        {"aten::add", op::translate_add},
        {"aten::add_", op::inplace_op<op::translate_add>},
        {"aten::addcmul", op::translate_addcmul},
        {"aten::addmm", op::translate_addmm},
        {"aten::asin", op::translate_1to1_match_1_inputs<opset8::Asin>},
        {"aten::asin_", op::inplace_op<op::translate_1to1_match_1_inputs<opset8::Asin>>},
        {"aten::asinh", op::translate_1to1_match_1_inputs<opset8::Asinh>},
        {"aten::asinh_", op::inplace_op<op::translate_1to1_match_1_inputs<opset8::Asinh>>},
        {"aten::as_tensor", op::translate_as_tensor},
        {"aten::atan", op::translate_1to1_match_1_inputs<opset8::Atan>},
        {"aten::atan_", op::inplace_op<op::translate_1to1_match_1_inputs<opset8::Atan>>},
        {"aten::atanh", op::translate_1to1_match_1_inputs<opset8::Atanh>},
        {"aten::atanh_", op::inplace_op<op::translate_1to1_match_1_inputs<opset8::Atanh>>},
        {"aten::avg_pool2d", op::translate_avg_pool2d},
        {"aten::batch_norm", op::translate_batch_norm},
        // {"aten::cat", done as transformation},
        {"aten::clamp", op::translate_clamp},
        {"aten::clamp_min", op::translate_1to1_match_2_inputs<opset8::Maximum>},
        {"aten::clamp_max", op::translate_1to1_match_2_inputs<opset8::Minimum>},
        {"aten::ceil", op::translate_1to1_match_1_inputs<opset8::Ceiling>},
        {"aten::ceil_", op::inplace_op<op::translate_1to1_match_1_inputs<opset8::Ceiling>>},
        {"aten::clone", op::skip_node},       // ignore clone operators that are inserted by PyTorch autograd
        {"aten::contiguous", op::skip_node},  // In openvino how tensors are stored in memory is internal plugin detail,
                                              // we assume all tensors are contiguous
        {"aten::conv2d", op::translate_conv2d},
        {"aten::convolution", op::translate_convolution},
        {"aten::cos", op::translate_1to1_match_1_inputs<opset8::Cos>},
        {"aten::cos_", op::inplace_op<op::translate_1to1_match_1_inputs<opset8::Cos>>},
        {"aten::cosh", op::translate_1to1_match_1_inputs<opset8::Cosh>},
        {"aten::cosh_", op::inplace_op<op::translate_1to1_match_1_inputs<opset8::Cosh>>},
        {"aten::cumsum", op::translate_1to1_match_2_inputs<opset8::CumSum>},
        {"aten::dim", op::translate_dim},
        {"aten::div", op::translate_div},
        {"aten::div_", op::inplace_op<op::translate_div>},
        {"aten::elu", op::translate_elu},
        {"aten::embedding", op::translate_embedding},
        {"aten::eq", op::translate_1to1_match_2_inputs<opset8::Equal>},
        {"aten::exp", op::translate_1to1_match_1_inputs<opset8::Exp>},
        {"aten::expand", op::translate_expand},
        {"aten::expand_as", op::translate_expand_as},
        {"aten::flatten", op::translate_flatten},
        {"aten::floor", op::translate_1to1_match_1_inputs<opset8::Floor>},
        {"aten::floor_", op::inplace_op<op::translate_1to1_match_1_inputs<opset8::Floor>>},
        {"aten::floordiv", op::translate_floordiv},
        {"aten::full", op::translate_full},
        {"aten::full_like", op::translate_full_like},
        {"aten::gelu", op::translate_gelu},
        {"aten::group_norm", op::translate_group_norm},
        {"aten::ge", op::translate_1to1_match_2_inputs<opset8::GreaterEqual>},
        {"aten::gt", op::translate_1to1_match_2_inputs<opset8::Greater>},
        {"aten::hardsigmoid", op::translate_1to1_match_1_inputs<opset8::HSigmoid>},
        {"aten::hardswish", op::translate_1to1_match_1_inputs<opset8::HSwish>},
        {"aten::hardswish_", op::inplace_op<op::translate_1to1_match_1_inputs<opset8::HSwish>>},
        {"aten::hardtanh", op::translate_hardtanh},
        {"aten::hardtanh_", op::inplace_op<op::translate_hardtanh>},
        {"aten::Int", op::translate_int},
        {"aten::im2col", op::translate_im2col},
        {"aten::is_grad_enabled", op::return_false_scalar},
        {"aten::layer_norm", op::translate_layer_norm},
        {"aten::leaky_relu", op::translate_1to1_match_2_inputs<opset8::PRelu>},
        {"aten::leaky_relu_", op::inplace_op<op::translate_1to1_match_2_inputs<opset8::PRelu>>},
        {"aten::linear", op::translate_linear},
        {"aten::le", op::translate_1to1_match_2_inputs<opset8::LessEqual>},
        {"aten::lt", op::translate_1to1_match_2_inputs<opset8::Less>},
        {"aten::matmul", op::translate_1to1_match_2_inputs<opset8::MatMul>},
        {"aten::masked_fill", op::translate_masked_fill},
        {"aten::masked_fill_", op::inplace_op<op::translate_masked_fill>},
        {"aten::max_pool2d", op::translate_max_pool2d},
        {"aten::max", op::translate_max},
        {"aten::mean", op::translate_mean},
        {"aten::min", op::translate_min},
        {"aten::mm", op::translate_1to1_match_2_inputs<opset8::MatMul>},
        {"aten::bmm", op::translate_1to1_match_2_inputs<opset8::MatMul>},
        {"aten::matmul", op::translate_1to1_match_2_inputs<opset8::MatMul>},
        {"aten::mul", op::translate_1to1_match_2_inputs<opset8::Multiply>},
        {"aten::mul_", op::inplace_op<op::translate_1to1_match_2_inputs<opset8::Multiply>>},
        {"aten::ne", op::translate_1to1_match_2_inputs<opset8::NotEqual>},
        {"aten::neg", op::translate_neg},
        {"aten::norm", op::translate_norm},
        {"aten::numel", op::translate_numel},
        {"aten::new_full", op::translate_new_full},
        {"aten::new_ones", op::translate_new_ones},
        {"aten::new_zeros", op::translate_new_zeros},
        {"aten::ones", op::translate_ones},
        {"aten::ones_like", op::translate_ones_like},
        {"aten::pad", op::translate_pad},
        {"aten::permute", op::translate_1to1_match_2_inputs<opset8::Transpose>},
        {"aten::pow", op::translate_1to1_match_2_inputs<opset8::Power>},
        {"aten::reciprocal", op::translate_reciprocal},
        {"aten::relu", op::translate_1to1_match_1_inputs<opset8::Relu>},
        {"aten::relu_", op::inplace_op<op::translate_1to1_match_1_inputs<opset8::Relu>>},
        {"aten::relu6", op::translate_relu6},
        {"aten::reshape", op::translate_reshape},
        {"aten::reshape_as", op::translate_reshape_as},
        {"aten::rsub", op::translate_rsub},
        {"aten::roll", op::translate_roll},
        {"aten::rsqrt", op::translate_rsqrt},
        {"aten::select", op::translate_select},
        {"aten::sigmoid", op::translate_1to1_match_1_inputs<opset8::Sigmoid>},
        {"aten::silu", op::translate_1to1_match_1_inputs<opset8::Swish>},
        {"aten::silu_", op::inplace_op<op::translate_1to1_match_1_inputs<opset8::Swish>>},
        {"aten::sin", op::translate_1to1_match_1_inputs<opset8::Sin>},
        {"aten::sin_", op::inplace_op<op::translate_1to1_match_1_inputs<opset8::Sin>>},
        {"aten::sinh", op::translate_1to1_match_1_inputs<opset8::Sinh>},
        {"aten::sinh_", op::inplace_op<op::translate_1to1_match_1_inputs<opset8::Sinh>>},
        {"aten::size", op::translate_size},
        {"aten::slice", op::translate_slice},
        {"aten::softmax", op::translate_softmax},
        {"aten::sqrt", op::translate_1to1_match_1_inputs<opset8::Sqrt>},
        {"aten::square", op::translate_square},
        {"aten::squeeze", op::translate_squeeze},
        {"aten::sub", op::translate_sub},
        {"aten::sum", op::translate_sum},
        {"aten::tan", op::translate_1to1_match_1_inputs<opset8::Tan>},
        {"aten::tan_", op::inplace_op<op::translate_1to1_match_1_inputs<opset8::Tan>>},
        {"aten::tanh", op::translate_1to1_match_1_inputs<opset8::Tanh>},
        {"aten::tanh_", op::inplace_op<op::translate_1to1_match_1_inputs<opset8::Tanh>>},
        {"aten::type_as",
         op::translate_1to1_match_2_inputs<opset8::ConvertLike>},  // TODO: overflow semantics is different
        {"aten::to", op::translate_to},
        {"aten::transpose", op::translate_transpose},
        {"aten::unfold", op::translate_unfold},
        {"aten::unsqueeze", op::translate_1to1_match_2_inputs<opset8::Unsqueeze>},
        {"aten::unsqueeze_", op::inplace_op<op::translate_1to1_match_2_inputs<opset8::Unsqueeze>>},
        {"aten::upsample_bicubic2d", op::translate_upsample_bicubic2d},
        {"aten::upsample_bilinear2d", op::translate_upsample_bilinear2d},
        {"aten::upsample_nearest2d", op::translate_upsample_nearest2d},
        {"aten::var", op::translate_var},
        {"aten::view", op::translate_view},
        {"aten::zeros", op::translate_zeros},
        {"aten::zeros_like", op::translate_zeros_like},
        {"prim::Constant", op::translate_constant},
        {"prim::If", op::translate_if},
        {"prim::is_cuda", op::return_false_scalar},
        {"prim::Loop", op::translate_loop},
        {"prim::NumToTensor", op::skip_node},  // In openvino we already store number as tensor with shape []
        {"prim::requires_grad", op::return_false_scalar},
        {"prim::TupleConstruct", op::translate_tuple_construct},
    };
};

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
