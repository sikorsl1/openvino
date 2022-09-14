#include "openvino/frontend/pytorch/node_context.hpp"

#include <openvino/frontend/exception.hpp>
#include <openvino/frontend/pytorch/decoder.hpp>
#include <openvino/opsets/opset8.hpp>

#include "exception.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

std::shared_ptr<opset8::Constant> NodeContext::get_constant_at_input(size_t index) const {
    OV_FRONTEND_REQUIRE(!input_is_none(index));
    auto input_node = get_input(index).get_node_shared_ptr();
    auto input = std::dynamic_pointer_cast<opset8::Constant>(input_node);
    FRONT_END_GENERAL_CHECK(input, "Input with index ", index, " cannot be interpretted as Constant: ", input_node);
    return input;
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
