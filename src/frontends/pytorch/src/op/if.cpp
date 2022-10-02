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

OutputVector translate_if(NodeContext& context) {
    auto if_node = std::make_shared<opset8::If>(context.get_input(0));
    context.mark_node(if_node);
    auto decoder = context.get_decoder();
    OV_FRONTEND_REQUIRE(decoder->get_subgraph_size() == 2);

    auto then_decoder = decoder->get_subgraph_decoder(0);
    auto then_body = context.convert_subgraph(0);
    if_node->set_then_body(then_body);
    auto then_inputs = then_decoder->inputs();

    auto else_decoder = decoder->get_subgraph_decoder(1);
    auto else_body = context.convert_subgraph(1);
    if_node->set_else_body(else_body);
    auto else_inputs = else_decoder->inputs();

    std::set<size_t> input_idxs;
    input_idxs.insert(then_inputs.begin(), then_inputs.end());
    input_idxs.insert(else_inputs.begin(), else_inputs.end());

    std::map<size_t, ParameterVector> inputs_map;
    std::map<size_t, ResultVector> outputs_map;
    for (auto param : then_body->get_parameters()) {
        auto name = param->get_output_tensor(0).get_any_name();
        size_t input_idx = (size_t)std::stoll(name);
        FRONT_END_OP_CONVERSION_CHECK(inputs_map.count(input_idx) == 0,
                                      "More then one then_body input with same tensor name: ",
                                      inputs_map.at(input_idx)[0],
                                      " adding: ",
                                      param);
        inputs_map[input_idx] = {param, nullptr};
    }
    for (auto param : else_body->get_parameters()) {
        auto name = param->get_output_tensor(0).get_any_name();
        size_t input_idx = (size_t)std::stoll(name);
        if (inputs_map.count(input_idx)) {
            inputs_map[input_idx][1] = param;
        } else {
            inputs_map[input_idx] = {nullptr, param};
        }
    }
    std::map<size_t, std::shared_ptr<opset8::Result>> then_body_results;
    std::map<size_t, std::shared_ptr<opset8::Result>> else_body_results;
    std::set<size_t> output_idxs;
    for (auto result : then_body->get_results()) {
        auto name = result->input(0).get_tensor().get_any_name();
        size_t output_idx = (size_t)std::stoll(name);
        FRONT_END_OP_CONVERSION_CHECK(then_body_results.count(output_idx) == 0,
                                      "More then one then_body output with same tensor name: ",
                                      then_body_results.at(output_idx),
                                      " adding: ",
                                      result);
        then_body_results[output_idx] = result;
        output_idxs.insert(output_idx);
    }
    for (auto result : else_body->get_results()) {
        auto name = result->input(0).get_tensor().get_any_name();
        size_t output_idx = (size_t)std::stoll(name);
        FRONT_END_OP_CONVERSION_CHECK(else_body_results.count(output_idx) == 0,
                                      "More then one then_body output with same tensor name: ",
                                      else_body_results.at(output_idx),
                                      " adding: ",
                                      result);
        then_body_results[output_idx] = result;
        output_idxs.insert(output_idx);
    }
    OutputVector res;
    for (int i = 0; i < context.num_of_outputs(); i++) {
        res.push_back(if_node->set_output(then_body->get_results()[i], else_body->get_results()[i]));
        OV_FRONTEND_REQUIRE(output_idxs.erase(then_decoder->output(i)));
        OV_FRONTEND_REQUIRE(output_idxs.erase(else_decoder->output(i)));
    }
    for (auto output_idx : output_idxs) {
        if (!then_body_results.count(output_idx)) {
            // Need to add Parameter->Result construction in then body
            auto new_parameter = std::make_shared<opset8::Parameter>(element::dynamic, PartialShape::dynamic());
            new_parameter->get_output_tensor(0).add_names({std::to_string(output_idx)});
            auto new_result = std::make_shared<opset8::Result>(new_parameter);
            then_body->add_parameters({new_parameter});
            then_body->add_results({new_result});
            then_body->validate_nodes_and_infer_types();
            FRONT_END_OP_CONVERSION_CHECK(inputs_map.count(output_idx), "Input must exist in else body");
            inputs_map[output_idx][0] = new_parameter;
            then_body_results[output_idx] = new_result;
            std::cout << "[ WARNING ] Modified then body: " << if_node << std::endl;
        } else if (!else_body_results.count(output_idx)) {
            // Need to add Parameter->Result construction in else body
            auto new_parameter = std::make_shared<opset8::Parameter>(element::dynamic, PartialShape::dynamic());
            new_parameter->get_output_tensor(0).add_names({std::to_string(output_idx)});
            auto new_result = std::make_shared<opset8::Result>(new_parameter);
            else_body->add_parameters({new_parameter});
            else_body->add_results({new_result});
            else_body->validate_nodes_and_infer_types();
            FRONT_END_OP_CONVERSION_CHECK(inputs_map.count(output_idx), "Input must exist in then body");
            inputs_map[output_idx][1] = new_parameter;
            else_body_results[output_idx] = new_result;
            std::cout << "[ WARNING ] Modified else body: " << if_node << std::endl;
        }
    }
    // Create prim::If inputs and outputs
    for (auto input : inputs_map) {
        if (!input_idxs.count(input.first)) {
            auto external_output = context.get_tensor_from_model_or_create_input(input.first);
            if_node->set_input(external_output, input.second[0], input.second[1]);
        } else {
            auto external_output = context.get_tensor_from_model(input.first);
            if (external_output.get_node()) {
                if_node->set_input(external_output, input.second[0], input.second[1]);
            }
        }
    }
    for (auto output_idx : output_idxs) {
        context.add_tensor_to_context(
            output_idx,
            if_node->set_output(then_body_results.at(output_idx), else_body_results.at(output_idx)));
    }
    if_node->validate_and_infer_types();
    return res;
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov