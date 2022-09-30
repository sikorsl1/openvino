#include "openvino/frontend/pytorch/node_context.hpp"

#include <openvino/frontend/exception.hpp>
#include <openvino/frontend/pytorch/decoder.hpp>
#include <openvino/opsets/opset8.hpp>

#include "exception.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

std::shared_ptr<opset8::Constant> NodeContext::get_constant_at_input(size_t index) const {
    OV_FRONTEND_REQUIRE(!input_is_none(index));
    auto input_node = get_input(index).get_node_shared_ptr();
    auto input = std::dynamic_pointer_cast<opset8::Constant>(input_node);
    auto param = std::dynamic_pointer_cast<opset8::Parameter>(input_node);
    if (!input && param) {
        // We need to look into external context for inputs that would be feed into this parameter
        auto name = param->get_output_tensor(0).get_any_name();
        size_t tensor_idx = (size_t)std::stoll(name);
        if (m_ext_tensor_map.count(tensor_idx)) {
            auto tensor = m_ext_tensor_map.at(tensor_idx);
            input_node = tensor.get_node_shared_ptr();
            input = std::dynamic_pointer_cast<opset8::Constant>(input_node);
        }
    }
    FRONT_END_GENERAL_CHECK(input, "Input with index ", index, " cannot be interpreted as Constant: ", input_node);
    return input;
}

std::shared_ptr<ov::Model> NodeContext::convert_subgraph(size_t index) {
    auto subgraph_decoder = m_decoder->get_subgraph_decoder(index);

    // Extend external context with internal tensors except Parameter nodes, because internal Parameters are created to
    // link internal context with external
    TensorMap ext_map(m_ext_tensor_map);
    for (auto tensor : *m_tensor_map) {
        auto node = tensor.second.get_node_shared_ptr();
        if (!std::dynamic_pointer_cast<opset8::Parameter>(node))
            ext_map[tensor.first] = tensor.second;
    }

    auto model = convert_pytorch_model(subgraph_decoder, ext_map);
    // Remove unused parameters, they could be created as inputs to the parts of graph that weren't
    // used for generating output.
    for (int i = subgraph_decoder->inputs().size(); i < model->get_parameters().size(); i++) {
        auto parameter = model->get_parameters()[i];
        if (parameter->output(0).get_target_inputs().empty()) {
            // There is no consumers: safe to remove
            //std::cout << "[ WARNING ] Removing parameter " << parameter
            //          << " in converted Pytorch model, because it is never used" << std::endl;
            model->remove_parameter(parameter);
        }
    }
    return model;
}

template <>
std::vector<int64_t> NodeContext::const_input<std::vector<int64_t>>(size_t index) const {
    return get_constant_at_input(index)->cast_vector<int64_t>();
}

template <>
std::string NodeContext::const_input<std::string>(size_t index) const {
    throw std::runtime_error("Cannot represent string as OV constant: lack of strings support");
    // return get_constant_at_input(index)->cast_vector<std::string>()[0];
}

template <>
ngraph::Strides NodeContext::const_input<ngraph::Strides>(size_t index) const {
    return get_constant_at_input(index)->cast_vector<ngraph::Strides::value_type>();
}

template <>
ngraph::CoordinateDiff NodeContext::const_input<ngraph::CoordinateDiff>(size_t index) const {
    return get_constant_at_input(index)->cast_vector<ngraph::CoordinateDiff::value_type>();
}

template <>
ngraph::Shape NodeContext::const_input<ngraph::Shape>(size_t index) const {
    return get_constant_at_input(index)->cast_vector<ngraph::Shape::value_type>();
}

template <>
int64_t NodeContext::const_input<int64_t>(size_t index) const {
    return get_constant_at_input(index)->cast_vector<int64_t>()[0];
}

template <>
bool NodeContext::const_input<bool>(size_t index) const {
    return get_constant_at_input(index)->cast_vector<bool>()[0];
}

template <>
double NodeContext::const_input<double>(size_t index) const {
    return get_constant_at_input(index)->cast_vector<double>()[0];
}

template <>
float NodeContext::const_input<float>(size_t index) const {
    return get_constant_at_input(index)->cast_vector<float>()[0];
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
