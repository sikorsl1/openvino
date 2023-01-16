// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <openvino/frontend/pytorch/decoder.hpp>
#include <openvino/frontend/pytorch/node_context.hpp>

#include "op_table.hpp"
#include "pt_framework_node.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

Output<Node> make_optional_bias(const Output<Node>& base_op,
                                const NodeContext& context,
                                size_t bias_input_idx,
                                const std::vector<int>& unsqueeze_dims) {
    using std::make_shared;

    if (!context.input_is_none(bias_input_idx)) {
        auto bias = context.get_input(bias_input_idx);
        if (!unsqueeze_dims.empty()) {
            auto indices = opset10::Constant::create(element::i32, {unsqueeze_dims.size()}, unsqueeze_dims);
            context.mark_node(indices);
            bias = make_shared<opset10::Unsqueeze>(bias, indices);
            context.mark_output(bias);
        }
        return make_shared<opset10::Add>(context.mark_output(base_op), bias);
    } else {
        return base_op;
    }
}

Output<ov::Node> reshape_conv_bias(NodeContext& context, Output<ov::Node> bias, Output<ov::Node> conv) {
    auto conv_shape = context.mark_node(std::make_shared<opset10::ShapeOf>(conv));
    auto conv_rank = context.mark_node(std::make_shared<opset10::ShapeOf>(conv_shape));
    auto one_const = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {1}));
    auto two_const = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {2}));
    auto tail_shape_rank = context.mark_node(std::make_shared<opset10::Subtract>(conv_rank, two_const));
    auto tail_shape = context.mark_node(std::make_shared<opset10::Broadcast>(one_const, tail_shape_rank));
    auto channels_dim = context.mark_node(std::make_shared<opset10::ShapeOf>(bias));
    auto new_shape =
        context.mark_node(std::make_shared<opset10::Concat>(OutputVector{one_const, channels_dim, tail_shape}, 0));

    return context.mark_node(std::make_shared<opset10::Reshape>(bias, new_shape, false));
}

std::shared_ptr<Node> get_rank_node(const Output<Node>& node) {
    auto shape = std::make_shared<opset10::ShapeOf>(node);
    return std::make_shared<opset10::ShapeOf>(shape);
}

Output<Node> reshape_kernel_for_group(const NodeContext& context,
                                      const Output<Node>& input,
                                      const Output<Node>& kernel,
                                      int64_t groups) {
    using std::make_shared;

    auto in_shape = std::make_shared<opset10::ShapeOf>(input);
    auto c_in_idx = opset10::Constant::create(element::i64, Shape{}, {1});
    auto axis_0 = opset10::Constant::create(element::i64, Shape{}, {0});
    auto in_shape_1 = make_shared<opset10::Gather>(in_shape, c_in_idx, axis_0);
    auto in_shape_1_uns = make_shared<opset10::Unsqueeze>(in_shape_1, axis_0);
    auto groups_const = opset10::Constant::create(element::i64, Shape{1}, {groups});
    auto c_in_value = make_shared<opset10::Divide>(in_shape_1_uns, groups_const);

    auto kernel_shape = std::make_shared<opset10::ShapeOf>(kernel);
    auto c_out_idx = opset10::Constant::create(element::i64, Shape{}, {0});
    auto kernel_shape_0 = make_shared<opset10::Gather>(kernel_shape, c_out_idx, axis_0);
    auto kernel_shape_0_uns = make_shared<opset10::Unsqueeze>(kernel_shape_0, axis_0);
    auto c_out_value = make_shared<opset10::Divide>(kernel_shape_0_uns, groups_const);

    auto start = opset10::Constant::create(element::i64, Shape{1}, {2});
    auto stop = opset10::Constant::create(element::i64, Shape{1}, {std::numeric_limits<int64_t>::max()});
    auto step = opset10::Constant::create(element::i64, Shape{1}, {1});
    auto remaining_shape = make_shared<opset10::Slice>(kernel_shape, start, stop, step);

    auto new_kernel_shape =
        make_shared<opset10::Concat>(OutputVector{groups_const, c_out_value, c_in_value, remaining_shape}, 0);
    context.mark_nodes({in_shape,
                        c_in_idx,
                        axis_0,
                        in_shape_1,
                        in_shape_1_uns,
                        groups_const,
                        c_in_value,
                        kernel_shape,
                        c_out_idx,
                        kernel_shape_0,
                        kernel_shape_0_uns,
                        c_out_value,
                        start,
                        stop,
                        step,
                        remaining_shape,
                        new_kernel_shape});
    return make_shared<opset10::Reshape>(kernel, new_kernel_shape, false);
}

std::shared_ptr<Node> get_axes_range(NodeContext& context, size_t input_id) {
    auto x = context.get_input(input_id);
    auto start = std::make_shared<opset10::Constant>(element::i32, Shape{}, 0);
    auto step = std::make_shared<opset10::Constant>(element::i32, Shape{}, 1);
    auto shape = context.mark_node(std::make_shared<opset10::ShapeOf>(x, element::i32));
    auto rank = context.mark_node(std::make_shared<opset10::ShapeOf>(shape, element::i32));
    auto reduced_rank = context.mark_node(std::make_shared<opset10::Squeeze>(rank));
    return context.mark_node(std::make_shared<opset10::Range>(start, reduced_rank, step, element::i32));
};

std::shared_ptr<Node> numel(NodeContext& context, size_t input_id) {
    auto x = context.get_input(input_id);
    auto input_shape = context.mark_node(std::make_shared<opset10::ShapeOf>(x));
    auto axes = context.mark_node(opset10::Constant::create(element::i64, Shape({1}), {0}));
    return context.mark_node(std::make_shared<opset10::ReduceProd>(input_shape, axes, false));
};

ov::element::Type convert_dtype(NodeContext& context, size_t input_id) {
    auto pt_type = context.const_input<int64_t>(input_id);
    FRONT_END_OP_CONVERSION_CHECK(TORCH_TO_OV_TYPE.count(pt_type), "Unknown type: ", pt_type);
    return TORCH_TO_OV_TYPE.at(pt_type);
};

std::shared_ptr<Node> concat_list_construct(std::shared_ptr<Node> input) {
    if (auto list_construct = cast_fw_node(input, "prim::ListConstruct")) {
        auto list_inputs = list_construct->input_values();
        OutputVector node_vector;
        auto zero = opset10::Constant::create(element::i32, Shape{}, {0});
        for (size_t i = 0; i < list_inputs.size(); i++) {
            auto node = concat_list_construct(list_inputs[i].get_node_shared_ptr());
            auto unsqueezed_node = std::make_shared<opset10::Unsqueeze>(node, zero);
            node_vector.push_back(unsqueezed_node);
        }
        return std::make_shared<opset10::Concat>(node_vector, 0);
    }
    return input;
}

OutputVector make_framework_node(NodeContext* context) {
    auto schema = context->get_schema();
    // TODO: properly process schema to get the actual position of mutable input
    // Hack. Can indicate mutable inputs, but can it be reliable?
    if (schema.find('!') != std::string::npos) {
        // We create additional output for such nodes. It contains new tensor that represents input that was changed.
        auto fw_node =
            std::make_shared<PtFrameworkNode>(context->get_decoder(), context->inputs(), context->num_of_outputs() + 1);
        fw_node->set_friendly_name(context->get_op_type());
        auto outputs = fw_node->outputs();
        // Usually mutated input index is 0, because it is usually "self" input, so we need to replace this tensor with
        // output we created.
        context->mutate_input(0, outputs.back());
        // std::cerr << "[ WARNING ] Created node with mutated 0 input. Schema: " << schema << std::endl;
        context->mark_node(fw_node);
        // For simplification we do not expect such operations to have extra bodies
        FRONT_END_OP_CONVERSION_CHECK(context->get_decoder()->get_subgraph_size() == 0,
                                      "Mutable operation has subgraphs.");
        return outputs;
    }

    // Pay attention to subgraphs that may appear in the node
    auto fw_node =
        std::make_shared<PtFrameworkNode>(context->get_decoder(), context->inputs(), context->num_of_outputs());
    fw_node->set_friendly_name(context->get_op_type());

    std::map<size_t, ParameterVector> inputs_map;
    std::map<size_t, ResultVector> extra_outputs_map;
    std::set<size_t> input_idxs;  // initial inputs
    // We need to remember initial inputs to be able to find extra inputs to body that were created to propagate
    // external context
    int num_body_outs = 0;
    for (size_t i = 0; i < context->get_decoder()->get_subgraph_size(); ++i) {
        auto subgraph_decoder = context->get_decoder()->get_subgraph_decoder(i);
        auto inputs = subgraph_decoder->inputs();
        input_idxs.insert(inputs.begin(), inputs.end());
        auto body = context->convert_subgraph(i);
        fw_node->set_function(i, body);
        for (const auto& param : body->get_parameters()) {
            auto name = param->get_output_tensor(0).get_any_name();
            size_t input_idx = (size_t)std::stoll(name);
            inputs_map[input_idx].push_back(param);
        }
        auto body_outputs = subgraph_decoder->outputs();
        if (i == 0) {
            num_body_outs = body_outputs.size();
        } else {
            FRONT_END_OP_CONVERSION_CHECK(
                num_body_outs == body_outputs.size(),
                "Number of outputs of this body is different from number of outputs of first body");
        }
        // Some bodies may have mutated inputs which we need to propagate to external context
        auto body_results = body->get_results();
        for (int i = num_body_outs; i < body_results.size(); i++) {
            auto name = body_results[i]->input(0).get_tensor().get_any_name();
            size_t out_idx = (size_t)std::stoll(name);
            FRONT_END_OP_CONVERSION_CHECK(extra_outputs_map.count(out_idx) == 0,
                                          "More then one body output with same tensor name.");
            extra_outputs_map[out_idx].push_back(body_results[i]);
        }
    }
    // Connect inputs with external context
    for (const auto& input : inputs_map) {
        if (!input_idxs.count(input.first)) {
            auto external_output = context->get_tensor_from_model_or_create_input(input.first);
            fw_node->set_invariant_inputs(external_output, input.second);
        } else {
            auto external_output = context->get_tensor_from_model(input.first);
            if (external_output.get_node()) {
                fw_node->set_invariant_inputs(external_output, input.second);
            }
        }
    }
    // Number of body outputs can be higher then number of pt node outputs, e.g. in case of loop first body output is
    // condition, we have to skip such outputs
    int num_skip_body_outputs =
        num_body_outs > context->num_of_outputs() ? num_body_outs - context->num_of_outputs() : 0;
    // We need to reduce number of outputs, because some outputs are outputs from body
    fw_node->set_output_size(context->num_of_outputs() - num_body_outs + num_skip_body_outputs);
    OutputVector res(context->mark_node(fw_node)->outputs());
    if (fw_node->get_internal_subgraphs_size() > 0) {
        auto first_body_results = fw_node->get_function(0)->get_results();
        std::vector<ResultVector> outputs;
        for (int i = num_skip_body_outputs; i < num_body_outs; i++) {
            outputs.push_back({first_body_results[i]});
        }
        for (int i = 1; i < fw_node->get_internal_subgraphs_size(); i++) {
            auto current_body_results = fw_node->get_function(i)->get_results();
            for (int i = num_skip_body_outputs; i < num_body_outs; i++) {
                outputs[i].push_back(current_body_results[i]);
            }
        }
        for (const auto& res_vec : outputs) {
            res.push_back(fw_node->set_body_outputs(res_vec));
        }
    }
    // Propagate extra outputs to external context
    for (const auto& output : extra_outputs_map) {
        context->add_tensor_to_context(output.first, fw_node->set_body_outputs(output.second));
    }
    return res;
}

OutputVector convert_node(NodeContext* context) {
    try {
        auto CONVERTERS_MAP = get_supported_ops();
        auto it = CONVERTERS_MAP.find(context->get_op_type());
        if (it != CONVERTERS_MAP.end()) {
            return it->second(*context);
        } /*else {
            const std::set<std::string> known_skips{"prim::RaiseException",
                                                    "aten::warn"};
            if (!known_skips.count(context->get_op_type())) {
                std::cout << "DIDN'T FIND converter for " << context->get_op_type() << " with inputs:";
                if (context->inputs().size() == 0) {
                    std::cout << " None";
                }
                for (auto input : context->inputs()) {
                    std::cout << " " << input;
                }
                std::cout << " with schema: " << context->get_schema() << std::endl;
            }
        }*/

    } catch (std::runtime_error& e) {
        std::cout << "Exception happened during conversion of op: " << context->get_op_type()
                  << " with schema: " << context->get_schema() << ": " << e.what() << '\n';
    } catch (...) {
        std::cout << "Some exception happened during conversion of node of type: " << context->get_op_type()
                  << std::endl;
    }
    // Create PtFrameworkNode for everything that wasn't able to be converted normally
    return make_framework_node(context);
}

/// \brief Completely convert pytorch_model, creates PtFrameworkNode if not possible to convert node
/// \param pytorch_model Input model
/// \param external_tensor_map Is used for recursive calls of convert_pytorch_model and represent the external context
///  which is visible from nested model. Empty external_tensor_map is used as an indication that this is a main body
///  conversion.
/// \return fully converted OV Model
std::shared_ptr<ov::Model> convert_pytorch_model2(std::shared_ptr<TorchDecoder> pytorch_model,
                                                  const TensorMap& external_tensor_map) {
    std::shared_ptr<ov::Model> resulting_model;  // define here to make a conversion in a nested scope
    {
        ParameterVector parameters;
        TensorMap tensor_map;  // tensor map of the current context
        std::set<size_t> mutated_tensors;

        //  Go over all pytorch_model inputs and register them in the tensor map:
        auto inputs = pytorch_model->inputs();
        for (int i = 0; i < inputs.size(); ++i) {
            PartialShape ps = pytorch_model->get_input_shape(i);
            auto type = simplified_type_interpret(pytorch_model->get_input_type(i));
            // TODO: Use special API to set custom type detalization
            auto parameter = std::make_shared<opset10::Parameter>(ov::element::dynamic, ps);
            parameter->get_output_tensor(0).add_names({std::to_string(pytorch_model->input(i))});
            parameters.push_back(parameter);
            auto order = pytorch_model->get_input_transpose_order(i);
            if (order.size() > 0 && !std::is_sorted(order.begin(), order.end())) {
                FRONT_END_GENERAL_CHECK(ps.is_static(), "Shape must be static.");  // TODO: make dynamic
                auto sh = ps.get_shape();
                Shape new_shape(sh.size());
                for (int i = 0; i < sh.size(); i++) {
                    new_shape[order[i]] = sh[i];
                }
                auto shape_const = opset10::Constant::create(element::i64, {new_shape.size()}, new_shape);
                auto reshape = std::make_shared<opset10::Reshape>(parameter, shape_const, false);
                auto order_const = opset10::Constant::create(element::i32, {order.size()}, order);
                auto transpose = std::make_shared<opset10::Transpose>(reshape, order_const);
                tensor_map[pytorch_model->input(i)] = transpose;
            } else {
                tensor_map[pytorch_model->input(i)] = parameter;
            }
        }

        auto node_visitor = [&](std::shared_ptr<TorchDecoder> node) {
            // Explore all inputs of node. Node may refer to input value that hasn't been created in the current scope.
            // But this value can be found in the outer scope, for this purpose we create new input for the model to
            // link with external scope on a higher level.

            auto raw_inputs = node->inputs();
            for (size_t i = 0; i < raw_inputs.size(); ++i) {
                auto input = node->input(i);
                if (tensor_map.find(input) == tensor_map.end()) {
                    // Input refers value in the outer scope, need to create a new Parameter in the current scope
                    // Linkage to external scope will be performed on the level of the parent operation (if or loop)
                    // TODO: Eliminate duplication with the main code for Parameters creation
                    PartialShape ps = node->get_input_shape(i);
                    auto type = simplified_type_interpret(node->get_input_type(i));
                    // TODO: Use special API to set custom type detalization
                    auto parameter = std::make_shared<opset10::Parameter>(element::dynamic, ps);
                    // TODO: Missing get_input_transpose_order handling for not trivial layouts
                    tensor_map[input] = parameter;
                    // set name of parameter to the index of node in the model
                    parameter->get_output_tensor(0).add_names({std::to_string(input)});
                    parameters.push_back(parameter);
                }
            }
            auto context = NodeContext(node, &tensor_map, &parameters, external_tensor_map);
            auto converted_outputs = convert_node(&context);

            auto mutated_t = context.get_mutated_tensors();
            mutated_tensors.insert(mutated_t.begin(), mutated_t.end());

            auto fw_outputs = node->outputs();
            // Ops with subgraphs or with mutated inputs may have more outputs after conversion compared to pytorch ones
            FRONT_END_OP_CONVERSION_CHECK(fw_outputs.size() <= converted_outputs.size(),
                                          "Number of ",
                                          node->get_op_type(),
                                          " outputs greater then number of converted outputs.");

            // TODO: Make sure that mapping of fw_outputs to converted_outputs does always work
            // FIXME: Now it is not true for at least prim::Constant
            for (size_t i = 0; i < fw_outputs.size(); ++i) {
                size_t fw_tensor_id = node->output(i);
                if (tensor_map.find(fw_tensor_id) != tensor_map.end()) {
                    throw std::runtime_error("Duplicated producer for PT value with unique ID: " +
                                             std::to_string(fw_tensor_id));
                }

                // Output shape of converted node should match the original output shape
                // OV_FRONTEND_REQUIRE(get_ov_shape(fw_outputs[i]) == converted_outputs[i].get_partial_shape());

                tensor_map[fw_tensor_id] = converted_outputs[i];
                converted_outputs[i].get_tensor().add_names({std::to_string(fw_tensor_id)});
            }
        };

        FRONT_END_GENERAL_CHECK(pytorch_model->get_subgraph_size() == 1, "Model should have exactly 1 subgraph.");
        pytorch_model->visit_subgraph(node_visitor);

        ResultVector results;
        for (size_t i = 0; i < pytorch_model->num_of_outputs(); ++i) {
            size_t id = pytorch_model->output(i);
            if (tensor_map.find(id) == tensor_map.end()) {
                // Not found in this scope, adding Parameter to connect to external scope
                auto parameter = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
                parameter->get_output_tensor(0).add_names({std::to_string(id)});
                parameters.push_back(parameter);
                tensor_map[id] = parameter;
            }
            auto ov_output = tensor_map[id];
            auto order = pytorch_model->get_output_transpose_order(i);
            FRONT_END_GENERAL_CHECK(order.size() == 0 || std::is_sorted(order.begin(), order.end()),
                                    "Output strides have wrong order.");
            FRONT_END_GENERAL_CHECK(ov_output.get_names().size() > 0,
                                    "Tensor doesn't have name, while it should have name: ",
                                    id);
            auto result = std::make_shared<opset10::Result>(ov_output);
            results.push_back(result);
        }

        // Since parameters can be added we need to list all current parameters
        std::set<size_t> param_names;
        for (const auto& param : parameters) {
            auto name = param->get_output_tensor(0).get_any_name();
            size_t input_idx = (size_t)std::stoll(name);
            param_names.insert(input_idx);
        }
        for (const auto& tensor_id : mutated_tensors) {
            if (param_names.count(tensor_id)) {
                FRONT_END_GENERAL_CHECK(tensor_map.count(tensor_id),
                                        "Tensor with id: ",
                                        tensor_id,
                                        " doesn't exist in tensor map.");
                // model input was mutated we need to make a result for it
                auto mutated_tensor = tensor_map.at(tensor_id);
                // empty external_tensor_map means this is main body of the model and we don't want to create
                // additional outputs in that case.
                if (mutated_tensor.get_target_inputs().empty() && !external_tensor_map.empty())
                    results.push_back(std::make_shared<opset10::Result>(tensor_map.at(tensor_id)));
            }
        }
        resulting_model = std::make_shared<ov::Model>(results, parameters);
        // Did a conversion in a nested scope to automatically remove any holders of nodes except those in the graph
    }

    return resulting_model;
}

std::shared_ptr<ov::op::util::FrameworkNode> cast_fw_node(std::shared_ptr<Node> node, const std::string& type) {
    auto fw_node = std::dynamic_pointer_cast<ov::op::util::FrameworkNode>(node);
    if (!fw_node) {
        return nullptr;
    }
    const auto& attrs = fw_node->get_attrs();
    if (attrs.find("PtTypeName") == attrs.end() || attrs.at("PtTypeName") != type) {
        return nullptr;
    }
    return fw_node;
}

Any simplified_type_interpret(Any type) {
    // Interpret Tensor[type] as just type
    // After applying of this interpretation we cannot distinguish true scalars (not tensors) and tensors with elements
    // of the same types
    if (type.is<type::Tensor>()) {
        auto tensor = type.as<type::Tensor>();
        if (tensor.element_type.is<element::Type>()) {
            return tensor.element_type;
        }
    }

    return type;
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
