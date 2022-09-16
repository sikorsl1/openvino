#include <map>
#include <exception>
#include <memory>
#include <limits>
#include <string>

#include "openvino/opsets/opset7.hpp"
#include <openvino/opsets/opset8.hpp>
#include <openvino/op/util/framework_node.hpp>
#include "openvino/frontend/node_context.hpp"

#include "openvino/frontend/pytorch/frontend.hpp"

#include "input_model.hpp"
#include "transforms.hpp"

namespace ov {
namespace frontend {
namespace pytorch {


#define OV_FRONTEND_REQUIRE(X) \
    do if(!(X)) { \
        throw std::runtime_error(std::string("[ ERROR ] Failed: ") + #X + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } while(false)


typedef std::map<size_t, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:

    NodeContext (std::shared_ptr<Decoder> decoder, const TensorMap& tensor_map) :
        // TODO: why the following ctor is explicit?
        frontend::NodeContext(decoder->get_op_type()),
        m_decoder(decoder),
        m_tensor_map(tensor_map) {}

    // Do not search for input in tensor map; try to access it as a constant of specified type T and return its value
    template <typename T>
    T const_input (size_t index) const;

    // Search for input in tensor map and return an output port for already converted op
    // TODO: int due to base class uses it, but naturally it should be size_t for PT
    Output<Node> get_input (int index) const override {
        //std::cerr << "Trying to map input to ngraph...";
        OV_FRONTEND_REQUIRE(!m_decoder->input_is_none(index));
        return m_tensor_map.at(m_decoder->input(index));
    }

    // TODO: upstream to base class
    OutputVector inputs () const {
        OutputVector res;
        for (size_t input : m_decoder->inputs()) {
            //std::cerr << "Searching for input: " << input->unique() << "\n";
            OV_FRONTEND_REQUIRE(m_tensor_map.find(input) != m_tensor_map.end());
            res.push_back(m_tensor_map.at(input));
        }
        return res;
    }

    bool input_is_none (size_t index) const {
        return m_decoder->input_is_none(index);
    }

    // Convert the resulting value of this node to ngraph Constant; works correctly only for nodes that produce
    // constant value, naturally for prim::Constant
    OutputVector as_constant () {
        return m_decoder->as_constant();
    }

    /*
    TODO: Should be uncommented when explicit NodeContext ctor won't require passing op_type
    const std::string& get_op_type() const override {
        return m_decoder->get_op_type();
    }
    */

    size_t num_of_outputs () const {
        return m_decoder->num_of_outputs();
    }

    std::vector<size_t> outputs () const {
        return m_decoder->outputs();
    }

    std::shared_ptr<Node> mark_node (std::shared_ptr<Node> ov_node) const  {
        return m_decoder->mark_node(ov_node);
    }

    void mark_nodes (std::vector<std::shared_ptr<Node>> ov_nodes) const {
        return m_decoder->mark_nodes(ov_nodes);
    }

    Output<Node> mark_output (Output<Node> ov_output) const {
        return m_decoder->mark_node(ov_output.get_node_shared_ptr());
    }

    Any get_attribute_as_any(const std::string&) const override {
        throw std::runtime_error("There is no any named attributes in Pytorch node, query by attribute name is not implemented");
    }

    void debug () const {
        m_decoder->debug();
    }

private:

    std::shared_ptr<opset8::Constant> get_constant_at_input (size_t index) const;

    std::shared_ptr<Decoder> m_decoder;
    const TensorMap& m_tensor_map;
};

std::shared_ptr<opset8::Constant> NodeContext::get_constant_at_input (size_t index) const {
    OV_FRONTEND_REQUIRE(!input_is_none(index));
    auto input = std::dynamic_pointer_cast<opset8::Constant>(get_input(index).get_node_shared_ptr());
    OV_FRONTEND_REQUIRE(input);
    return input;
}

template <>
std::vector<int64_t> NodeContext::const_input<std::vector<int64_t>> (size_t index) const {
    return get_constant_at_input(index)->cast_vector<int64_t>();
}

template <>
std::string NodeContext::const_input<std::string> (size_t index) const {
    throw std::runtime_error("Cannot represent string as OV constant: lack of strings support");
    //return get_constant_at_input(index)->cast_vector<std::string>()[0];
}

template <>
ngraph::Strides NodeContext::const_input<ngraph::Strides> (size_t index) const {
    return get_constant_at_input(index)->cast_vector<ngraph::Strides::value_type>();
}

template <>
ngraph::CoordinateDiff NodeContext::const_input<ngraph::CoordinateDiff> (size_t index) const {
    return get_constant_at_input(index)->cast_vector<ngraph::CoordinateDiff::value_type>();
}

template <>
ngraph::Shape NodeContext::const_input<ngraph::Shape> (size_t index) const {
    return get_constant_at_input(index)->cast_vector<ngraph::Shape::value_type>();
}

template <>
int64_t NodeContext::const_input<int64_t> (size_t index) const {
    return get_constant_at_input(index)->cast_vector<int64_t>()[0];
}

template <>
bool NodeContext::const_input<bool> (size_t index) const {
    return get_constant_at_input(index)->cast_vector<bool>()[0];
}

template <>
double NodeContext::const_input<double> (size_t index) const {
    return get_constant_at_input(index)->cast_vector<double>()[0];
}

template <>
float NodeContext::const_input<float> (size_t index) const {
    return get_constant_at_input(index)->cast_vector<float>()[0];
}


class PtFrameworkNode : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("PtFrameworkNode", "util", ::ov::op::util::FrameworkNode);

    PtFrameworkNode(const std::shared_ptr<Decoder>& decoder, const OutputVector& inputs)
        : ov::op::util::FrameworkNode(inputs, decoder->num_of_outputs()),
          m_decoder(decoder) {
        ov::op::util::FrameworkNodeAttrs attrs;
        //std::cerr << "[ DEBUG ] Making  PtFrameworkNode for " << m_decoder->get_op_type() << "\n";
        attrs.set_type_name("PTFrameworkNode");
        attrs["PtTypeName"] = m_decoder->get_op_type();
        set_attrs(attrs);

        //std::cout << attrs["PtTypeName"] << std::endl;

        // Set output shapes and types if recognized

        for(size_t i = 0; i < m_decoder->num_of_outputs(); ++i) {
            PartialShape ps;
            // TODO: Try to decode PT type as a custom type
            Any type = element::dynamic;
            // FIXME: ROUGH
            try {
                ps = m_decoder->get_output_shape(i);
            } catch (...) {
                // nothing, means the info cannot be queried and remains unknown
            }
            // FIXME: ROUGH
            try {
                type = m_decoder->get_output_type(i);
            } catch (std::runtime_error& e) {
                // nothing, means the info cannot be queried and remains unknown
                std::cerr << "[ ERROR ] Cannot retrieve type\n" << e.what() << std::endl;
            }
            // Let's see what type we have
            //std::cout << "Can be represented as element::Type: " << type.is<element::Type>() << std::endl;
            //std::cout << "element::Type value: " << type.as<element::Type>() << "\n";
            //std::exit(0);
            set_custom_output_type(i, type, ps);
        }
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return std::make_shared<PtFrameworkNode>(m_decoder, inputs);
    }

    std::string get_op_type() const {
        return m_decoder->get_op_type();
    }

    Decoder* get_decoder() const {
        return m_decoder.get();
    }

    void add_subgraph (std::shared_ptr<ov::Model> subgraph) {
        m_subgraphs.push_back(subgraph);
    }

    bool visit_attributes(AttributeVisitor& visitor) override {
        bool parent_visit_result = FrameworkNode::visit_attributes(visitor);
        for(size_t i = 0; i < m_subgraphs.size(); ++i) {
            std::string name = "subgraph_" + std::to_string(i);
            visitor.on_attribute(name, m_subgraphs[i]);
        }
        return parent_visit_result;
    }

private:
    std::shared_ptr<Decoder> m_decoder;
    std::vector<std::shared_ptr<ov::Model>> m_subgraphs;
};


Output<Node> make_optional_bias(
        Output<Node> base_op,
        const NodeContext& context,
        size_t bias_input_idx,
        std::vector<int> unsqueeze_dims = {})
{
    using namespace ngraph;
    using std::make_shared;

    if(!context.input_is_none(bias_input_idx)) {
        auto bias = context.get_input(bias_input_idx);
        if(!unsqueeze_dims.empty()) {
            auto indices = opset7::Constant::create(element::i32, { unsqueeze_dims.size() }, unsqueeze_dims);
            context.mark_node(indices);
            bias = make_shared<opset7::Unsqueeze>(bias, indices);
            context.mark_output(bias);
        }
        return make_shared<opset7::Add>(context.mark_output(base_op), bias);
    } else {
        return base_op;
    }
}

std::shared_ptr<ov::Node> get_rank_node(ov::Output<ov::Node> node) {
    auto shape = std::make_shared<opset8::ShapeOf>(node);
    return std::make_shared<opset8::ShapeOf>(shape);
}

Output<Node> reshape_kernel_for_group(
        const NodeContext& context,
        Output<Node> input,
        Output<Node> kernel,
        int64_t groups)
{
    using namespace ngraph;
    using std::make_shared;

    auto in_shape = std::make_shared<opset8::ShapeOf>(input);
    auto c_in_idx = opset8::Constant::create(element::i64, Shape{}, {1});
    auto axis_0 = opset8::Constant::create(element::i64, Shape{}, {0});
    auto in_shape_1 = make_shared<opset8::Gather>(in_shape, c_in_idx, axis_0);
    auto in_shape_1_uns = make_shared<opset8::Unsqueeze>(in_shape_1, axis_0);
    auto groups_const = opset8::Constant::create(element::i64, Shape{1}, {groups});
    auto c_in_value = make_shared<opset8::Divide>(in_shape_1_uns, groups_const);

    auto kernel_shape = std::make_shared<opset8::ShapeOf>(kernel);
    auto c_out_idx = opset8::Constant::create(element::i64, Shape{}, {0});
    auto kernel_shape_0 = make_shared<opset8::Gather>(kernel_shape, c_out_idx, axis_0);
    auto kernel_shape_0_uns = make_shared<opset8::Unsqueeze>(kernel_shape_0, axis_0);
    auto c_out_value = make_shared<opset8::Divide>(kernel_shape_0_uns, groups_const);

    auto start = opset8::Constant::create(element::i64, Shape{1}, {2});
    auto stop = opset8::Constant::create(element::i64, Shape{1}, {std::numeric_limits<int64_t>::max()});
    auto step = opset8::Constant::create(element::i64, Shape{1}, {1});
    auto remaining_shape = make_shared<opset8::Slice>(kernel_shape, start, stop, step);

    auto new_kernel_shape = make_shared<opset8::Concat>(OutputVector{groups_const, c_out_value, c_in_value, remaining_shape}, 0);
    context.mark_nodes({in_shape, c_in_idx, axis_0, in_shape_1, in_shape_1_uns, groups_const, c_in_value,
                        kernel_shape, c_out_idx, kernel_shape_0, kernel_shape_0_uns, c_out_value, start, stop, step, remaining_shape, new_kernel_shape});
    return make_shared<opset8::Reshape>(kernel, new_kernel_shape, false);
}

// TODO: Remove this super-hack, tensor_map should be local for each conversion activity
// Now it is global to simplify passing it to along the call stack


std::shared_ptr<ov::Model> convert_pytorch_model(std::shared_ptr<Decoder> pytorch_model, const TensorMap& external_tensor_map);

OutputVector convert_node(const std::shared_ptr<Decoder> decoder, TensorMap& tensor_map) {
    //std::cout << "[  ----  DEBUG  ---- ] convert_node\n";
    using ints = std::vector<int64_t>;
    using namespace ngraph;
    using std::make_shared;

    //std::cerr << "---\nAttempting to convert " << node->kind().toQualString() << "\n";
    //node->dump();


    auto context = NodeContext(decoder, tensor_map);

    //std::cerr << "[ DEBUG ] Attempting to convert " << context.get_op_type() << "\n";

        
    auto relu = [&]() -> OutputVector {
        return { context.mark_node(make_shared<opset7::Relu>(context.get_input(0))) };
    };

    // TODO: there is also the 3rd input for add in some cases, involve it to the conversion
    auto add = [&]() -> OutputVector {
        return { context.mark_node(make_shared<opset7::Add>(context.get_input(0), context.get_input(1))) };
    };

    try {
    std::map<std::string, std::function<OutputVector()> > converters = {

        { "aten::relu", relu},

        // TODO: Handle inplace semantics correctly
        { "aten::relu_", relu}, // inplace version of relu, for us it is identical to regular relu

        { "aten::conv2d", [&]() -> OutputVector {
            auto strides = context.const_input<Strides>(3);
            auto pads_begin = context.const_input<CoordinateDiff>(4);  // FIXME: The same 4 is used twice
            auto pads_end = context.const_input<CoordinateDiff>(4);    // FIXME: The same 4 is used twice
            auto dilations = context.const_input<Strides>(5);
            auto groups = context.const_input<int64_t>(6);

            std::shared_ptr<ov::Node> conv;
            if (groups == 1) {
                conv = make_shared<opset7::Convolution>(
                    context.get_input(0),
                    context.get_input(1),
                    strides,
                    pads_begin,
                    pads_end,
                    dilations
                );
            } else {
                conv = make_shared<opset7::GroupConvolution>(
                    context.get_input(0),
                    reshape_kernel_for_group(context, context.get_input(0), context.get_input(1), groups),
                    strides,
                    pads_begin,
                    pads_end,
                    dilations
                );
            }

            // FIXME: Doesn't work for dynamic rank
            // FIXME: Works for 2D convolutions only
            return { context.mark_output(make_optional_bias(conv, context, 2, {-2, -1})) };
        }},

        { "aten::_convolution", [&]() -> OutputVector {
            bool transposed = context.const_input<bool>(6);
            // TODO: Handle this temporary limitation
            OV_FRONTEND_REQUIRE(!transposed);

            auto strides = context.const_input<Strides>(3);
            auto pads_begin = context.const_input<CoordinateDiff>(4);  // FIXME: The same 4 is used twice
            auto pads_end = context.const_input<CoordinateDiff>(4);    // FIXME: The same 4 is used twice
            auto dilations = context.const_input<Strides>(5);
            // TODO: Handle skipped input 7 (6 was used above) -- what is it for?
            auto groups = context.const_input<int64_t>(8);

            std::shared_ptr<ov::Node> conv;
            if (groups == 1) {
                conv = make_shared<opset7::Convolution>(
                    context.get_input(0),
                    context.get_input(1),
                    strides,
                    pads_begin,
                    pads_end,
                    dilations
                );
            } else {
                conv = make_shared<opset7::GroupConvolution>(
                    context.get_input(0),
                    context.mark_output(reshape_kernel_for_group(context, context.get_input(0), context.get_input(1), groups)),
                    strides,
                    pads_begin,
                    pads_end,
                    dilations
                );
            }

            // FIXME: Doesn't work for dynamic rank -- why? It supports only 2D convs, but rank value can be not known in static
            // FIXME: Works for 2D convolutions only
            return { context.mark_output(make_optional_bias(conv, context, 2, {-2, -1})) };
        }},

        { "aten::batch_norm", [&]() -> OutputVector {
            auto training = context.const_input<bool>(5);
            OV_FRONTEND_REQUIRE(!training);   // TODO: support bn training
            return { context.mark_node(make_shared<opset7::BatchNormInference>(
                        context.get_input(0),
                        context.get_input(1),
                        context.get_input(2),
                        context.get_input(3),
                        context.get_input(4),
                        context.const_input<float>(7)  // epsilon
                    )) };
        }},

        {"aten::layer_norm", [&]() -> OutputVector {
            auto normalized_shape = context.const_input<Shape>(1);
            auto in_pshape_last_dim = *context.get_input(0).get_partial_shape().rbegin();
            OV_FRONTEND_REQUIRE(
                normalized_shape.size() == 1 && in_pshape_last_dim.is_static() &&
                static_cast<uint64_t>(in_pshape_last_dim.get_length()) == normalized_shape.back());
            auto eps = context.const_input<float>(4);
            auto axes = context.mark_node(opset7::Constant::create(element::i64, Shape{1}, {-1})); // TODO: support any dimention
            auto mvn = context.mark_node(make_shared<opset7::MVN>(context.get_input(0), axes, true, eps, op::MVNEpsMode::INSIDE_SQRT));
            std::shared_ptr<ov::Node> out_node = std::dynamic_pointer_cast<ov::Node>(mvn);
            if (!context.input_is_none(2)) {
                auto mul = make_shared<opset7::Multiply>(out_node, context.get_input(2));
                out_node = std::dynamic_pointer_cast<ov::Node>(mul);
            }
            if (!context.input_is_none(3)) {
                auto add = make_shared<opset7::Add>(out_node, context.get_input(3));
                out_node = std::dynamic_pointer_cast<ov::Node>(add);
            }
            return { context.mark_node(out_node) };
        }},

        { "aten::add", add},

        // TODO: Handle inplace semantics correctly
        { "aten::add_", add},   // inplace version of add, for us it is identical to regular add

        { "aten::mul", [&]() -> OutputVector {
            return { context.mark_node(make_shared<opset7::Multiply>(context.get_input(0), context.get_input(1))) };
        }},

        { "aten::div", [&]() -> OutputVector {
            auto pythondiv = false;
            if (!context.input_is_none(2)) {
                auto rounding_mode = context.const_input<std::string>(2);
                if (rounding_mode == "floor") {
                    pythondiv = true;
                } else if (rounding_mode == "trunc") {
                    pythondiv = true;
                    //break;
                }
            }
            return { context.mark_node(make_shared<opset7::Divide>(context.get_input(0), context.get_input(1), pythondiv)) };
        }},

        { "aten::tanh", [&]() -> OutputVector {
            return { context.mark_node(make_shared<opset7::Tanh>(context.get_input(0))) };
        }},

        { "aten::elu", [&]() -> OutputVector {
            auto alpha = context.const_input<float>(1);
            return { context.mark_node(make_shared<opset7::Elu>(context.get_input(0), alpha)) };
        }},

        { "aten::sigmoid", [&]() -> OutputVector {
            return { context.mark_node(make_shared<opset7::Sigmoid>(context.get_input(0))) };
        }},

        { "aten::silu_", [&]() -> OutputVector {
            // TODO: Handle inplace semantics correctly
            auto sigmoid = context.mark_node(make_shared<opset7::Sigmoid>(context.get_input(0)));
            auto silu = context.mark_node(make_shared<opset7::Multiply>(context.get_input(0), sigmoid));
            return { silu };
        }},

        { "aten::gelu", [&]() -> OutputVector {
            return { context.mark_node(make_shared<opset7::Gelu>(context.get_input(0))) };
        }},

        { "aten::sqrt", [&]() -> OutputVector {
            return { context.mark_node(make_shared<opset7::Sqrt>(context.get_input(0))) };
        }},

        { "aten::abs", [&]() -> OutputVector {
            return { context.mark_node(make_shared<opset7::Abs>(context.get_input(0))) };
        }},

        { "aten::square", [&]() -> OutputVector {
            auto input_0 = context.get_input(0);
            auto const_2 = context.mark_node(opset7::Constant::create(input_0.get_element_type(), Shape{1}, {2}));
            return { context.mark_node(make_shared<opset7::Power>(input_0, const_2)) };
        }},

        { "aten::hardtanh", [&]() -> OutputVector {
            auto min = context.const_input<float>(1);
            auto max = context.const_input<float>(2);
            return { context.mark_node(make_shared<opset7::Clamp>(context.get_input(0), min, max)) };
        }},

        { "aten::hardsigmoid", [&]() -> OutputVector {
            return { context.mark_node(make_shared<opset7::HSigmoid>(context.get_input(0))) };
        }},

        { "aten::hardswish", [&]() -> OutputVector {
            return { context.mark_node(make_shared<opset7::HSwish>(context.get_input(0))) };
        }},

        { "aten::relu6", [&]() -> OutputVector {
            return { context.mark_node(make_shared<opset7::Clamp>(context.get_input(0), 0., 6.)) };
        }},

        { "aten::softmax", [&]() -> OutputVector {
            auto axis = context.const_input<int64_t>(1);
            if (axis < 0) {
                auto in_rank = context.get_input(0).get_partial_shape().rank();
                OV_FRONTEND_REQUIRE(in_rank.is_static());
                axis = in_rank.get_length() + axis;
            }
            return { context.mark_node(make_shared<opset7::Softmax>(context.get_input(0), static_cast<size_t>(axis))) };
        }},

        { "aten::cat", [&]() -> OutputVector {
          // aten::cat needs a special handling since it takes a Tensor[] as
          // input. We set the inputs of ListConstruct as the inputs of cat.
          //
          // Pytorch IR:                              LLGA sees:
          //     %a    %b     %c          %dim              %a    %b    %c
          //      \     |     /             |                \     |    /
          //   prim::ListConstruct   prim::Constant     llga::Concat[axis=%dim]
          //                    \      /
          //                    aten::cat
          auto listConstruct = context.get_input(0).get_node();
          auto listConstruct_fw_node = dynamic_cast<PtFrameworkNode*>(listConstruct);
          OV_FRONTEND_REQUIRE(listConstruct_fw_node);
          OV_FRONTEND_REQUIRE(listConstruct_fw_node->get_decoder()->get_op_type() == "prim::ListConstruct");
          std::cerr << "POINT 1\n";
          auto axis = context.const_input<int64_t>(1);
          std::cerr << "POINT 2\n";
          OutputVector inputs;
          for (auto& input : listConstruct->inputs()) {
            inputs.push_back(input.get_source_output());
          }
          std::cerr << "POINT 3\n";
          auto result = context.mark_node(make_shared<opset7::Concat>(inputs, axis));
          //auto list_set = listConstruct_fw_node->get_rt_info()["pt_node"].as<std::set<const Node*>>();   // TODO: fails if marking doesn't really mark anything
          std::cerr << "POINT 4\n";
          //result->get_rt_info()["pt_node"].as<std::set<const Node*>>().insert(list_set.begin(), list_set.end());
          std::cerr << "POINT 5\n";
          return { result };
        }},

        { "aten::matmul", [&]() -> OutputVector {
            return { context.mark_node(make_shared<opset7::MatMul>(context.get_input(0), context.get_input(1))) };
        }},

        { "aten::mm", [&]() -> OutputVector {
            return { context.mark_node(make_shared<opset7::MatMul>(context.get_input(0), context.get_input(1))) };
        }},

        { "aten::linear", [&]() -> OutputVector {
            auto matmul = make_shared<opset7::MatMul>(context.get_input(0), context.get_input(1), false, true);
            return { context.mark_output(make_optional_bias(matmul, context, 2)) };
        }},

        { "aten::max_pool2d", [&]() -> OutputVector {
            auto kernel = context.const_input<Shape>(1);
            auto strides = context.const_input<Strides>(2);
            auto pads_begin = context.const_input<Shape>(3);  // FIXME: The same 3 is used twice
            auto pads_end = context.const_input<Shape>(3);    // FIXME: The same 3 is used twice
            auto dilations = context.const_input<Strides>(4);
            auto rounding_type = context.const_input<bool>(5) ? op::RoundingType::CEIL : op::RoundingType::FLOOR;

            // TODO: Upgrade to opset8::MaxPool to use dilations; for now we suppose they are all zeros
            return { context.mark_node(make_shared<opset7::MaxPool>(
                    context.get_input(0), strides, /*dilations,*/ pads_begin, pads_end, kernel, rounding_type)) };
        }},

        { "aten::avg_pool2d", [&]() -> OutputVector {
            auto kernel = context.const_input<Shape>(1);
            auto strides = context.const_input<Strides>(2);
            auto pads_begin = context.const_input<Shape>(3);  // FIXME: The same 3 is used twice
            auto pads_end = context.const_input<Shape>(3);    // FIXME: The same 3 is used twice
            auto rounding_type = context.const_input<bool>(4) ? op::RoundingType::CEIL : op::RoundingType::FLOOR;
            auto exclude_pad = !context.const_input<bool>(5);
            // TODO: support divisor override
            // auto divisor_override = context.const_input<int64_t>(6);

            return { context.mark_node(make_shared<opset7::AvgPool>(
                    context.get_input(0), strides, pads_begin, pads_end, kernel, exclude_pad, rounding_type)) };
        }},

        { "aten::adaptive_avg_pool2d", [&]() -> OutputVector {
            return { context.mark_node(make_shared<opset8::AdaptiveAvgPool>(context.get_input(0), context.get_input(1))) };
        }},

        { "aten::adaptive_max_pool2d", [&]() -> OutputVector {
            auto adaptive_max_pool = context.mark_node(make_shared<opset8::AdaptiveMaxPool>(context.get_input(0), context.get_input(1)));
            auto return_indices = context.const_input<bool>(2);
            OutputVector res{adaptive_max_pool->output(0)};
            if (return_indices) {
                res.push_back(adaptive_max_pool->output(1));
            }
            return res;
        }},

        { "aten::mean", [&]() -> OutputVector {
            auto keep_dims = context.const_input<bool>(2);            
            OV_FRONTEND_REQUIRE(context.input_is_none(3));
            return { context.mark_node(make_shared<opset8::ReduceMean>(context.get_input(0), context.get_input(1), keep_dims)) };
        }},

        { "aten::flatten", [&]() -> OutputVector {
            auto start_dim = context.const_input<int64_t>(1);
            auto end_dim = context.const_input<int64_t>(2);
            auto data_pshape = context.get_input(0).get_partial_shape();
            OV_FRONTEND_REQUIRE(data_pshape.rank().is_static()); // TODO: support dynamic rank
            auto rank = data_pshape.rank().get_length();
            if (start_dim < 0) {
                start_dim = rank + start_dim;
            }
            if (end_dim < 0) {
                end_dim = rank + end_dim;
            }
            OV_FRONTEND_REQUIRE(start_dim < end_dim);
            auto delta = end_dim - start_dim;
            std::vector<int64_t> new_shape(rank - delta, 0);
            new_shape[start_dim] = -1;
            auto new_shape_const = context.mark_node(opset7::Constant::create(element::i64, {new_shape.size()}, new_shape));
            return { context.mark_node(make_shared<opset8::Reshape>(context.get_input(0), new_shape_const, true)) };
        }},

        { "prim::NumToTensor", [&]() -> OutputVector {
            // Do nothing // TODO: Really? Should we produce scalar tensor with shape [] instead of custom PT type?
            return { context.mark_node(context.get_input(0).get_node_shared_ptr()) };
        }},

        { "aten::contiguous", [&]() -> OutputVector {
            // Do nothing
            return { context.mark_node(context.get_input(0).get_node_shared_ptr()) };
        }},

        { "aten::as_tensor", [&]() -> OutputVector {
            OV_FRONTEND_REQUIRE(context.const_input<int64_t>(1) == 6);
            OV_FRONTEND_REQUIRE(context.input_is_none(2));
            //auto new_shape_const = context.mark_node(opset7::Constant::create(element::i64, {1}, {1}));
            //return { context.mark_node(std::make_shared<opset8::Reshape>(context.get_input(0), new_shape_const->output(0), true)) };
            return { context.mark_output(context.get_input(0)) };
        }},

        { "aten::Int", [&]() -> OutputVector {
            return { context.mark_node(make_shared<opset8::Convert>(context.get_input(0), element::i64)) };
        }},

        { "aten::to", [&]() -> OutputVector {
            auto dtype = element::f32;
            // TODO: figure out all inputs meaning
            OV_FRONTEND_REQUIRE(context.const_input<int64_t>(1) == 6);
            OV_FRONTEND_REQUIRE(context.const_input<bool>(2) == false);
            OV_FRONTEND_REQUIRE(context.const_input<bool>(3) == false);
            OV_FRONTEND_REQUIRE(context.input_is_none(4));
            return { context.mark_node(make_shared<opset8::Convert>(context.get_input(0), dtype)) };
        }},

        { "aten::permute", [&]() -> OutputVector {
            return { context.mark_node(make_shared<opset7::Transpose>(context.get_input(0), context.get_input(1))) };
        }},

        { "aten::embedding", [&]() -> OutputVector {
            // TODO: find out the meaning of input idx 2
            OV_FRONTEND_REQUIRE(context.const_input<bool>(3) == false);
            OV_FRONTEND_REQUIRE(context.const_input<bool>(4) == false);
            auto axis_0 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
            return { context.mark_node(make_shared<opset7::Gather>(context.get_input(0), context.get_input(1), axis_0)) };
        }},

        { "aten::transpose", [&]() -> OutputVector {
            auto dim0 = context.const_input<int64_t>(1);
            auto dim1 = context.const_input<int64_t>(2);
            auto data_pshape = context.get_input(0).get_partial_shape();
            auto rank = data_pshape.rank();
            OV_FRONTEND_REQUIRE(rank.is_static());
            auto _rank = rank.get_length();
            if (dim0 < 0) {
                dim0 = _rank + dim0;
            }
            if (dim1 < 0) {
                dim1 = _rank + dim1;
            }
            OV_FRONTEND_REQUIRE(dim0 > 0 && dim1 > 0);
            OV_FRONTEND_REQUIRE(dim0 < _rank && dim1 < _rank);
            std::vector<int64_t> order(_rank, 0);
            std::iota(order.begin(), order.end(), 0);
            std::swap(order[dim0], order[dim1]);
            auto order_const = context.mark_node(opset7::Constant::create(element::i64, {order.size()}, order));
            return { context.mark_node(make_shared<opset7::Transpose>(context.get_input(0), order_const)) };
        }},

        { "aten::size", [&]() -> OutputVector {
            auto shape = context.mark_node(make_shared<opset8::ShapeOf>(context.get_input(0), element::i32));
            if(context.input_is_none(1)) {
                return shape->outputs();
            } else {
                auto axis_0 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
                return { context.mark_node(make_shared<opset8::Gather>(shape, context.get_input(1), axis_0)) };
            }
        }},

        { "aten::view", [&]() -> OutputVector {
            auto shape_node = context.get_input(1).get_node();
            auto shape_node_fw_node = dynamic_cast<PtFrameworkNode*>(shape_node);
            std::shared_ptr<ov::Node> reshape;
            if (shape_node_fw_node && shape_node_fw_node->get_decoder()->get_op_type() == "prim::ListConstruct") {
                // TODO: maybe use pt shape instead of whole shape subgraph, because it may be more efficent
                OutputVector inputs;
                auto axis_0 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
                for (auto& input : shape_node->inputs()) {
                    auto rank = input.get_partial_shape().rank();
                    OV_FRONTEND_REQUIRE(rank.is_dynamic() || rank.get_length() == 0);
                    auto unsqueeze = context.mark_node(make_shared<opset7::Unsqueeze>(input.get_source_output(), axis_0));
                    inputs.push_back(unsqueeze);
                }
                auto concat = context.mark_node(make_shared<opset7::Concat>(inputs, 0));
                reshape = context.mark_node(make_shared<opset7::Reshape>(context.get_input(0), concat, false));
                // TODO: temporary disabled
                //auto list_set = shape_node_fw_node->get_rt_info()["pt_node"].as<std::set<const Node*>>();
                //reshape->get_rt_info()["pt_node"].as<std::set<const Node*>>().insert(list_set.begin(), list_set.end());
            } else {
                reshape = context.mark_node(make_shared<opset7::Reshape>(context.get_input(0), context.get_input(1), false));
            }
            return { reshape };
        }},

        { "prim::ListUnpack", [&]() -> OutputVector {
            auto split_with_sizes = dynamic_cast<PtFrameworkNode*>(context.get_input(0).get_node());
            if (split_with_sizes && split_with_sizes->get_decoder()->get_op_type() == "aten::split_with_sizes") {
                auto split = make_shared<opset7::VariadicSplit>(
                    split_with_sizes->get_input_source_output(0),
                    split_with_sizes->get_input_source_output(2),
                    split_with_sizes->get_input_source_output(1));
                return context.mark_node(split)->outputs();
            }
            throw std::runtime_error("Cannot match prim::ListUnpack with expected aten::split_with_sizes as an input, left prim::ListUnpack not converted");
        }},

        { "aten::unsqueeze", [&]() -> OutputVector {
            return { context.mark_node(make_shared<opset8::Unsqueeze>(context.get_input(0), context.get_input(1))) };
        }},

        { "aten::rsub", [&]() -> OutputVector {
            // reverse aten::sub other - self * alpha
            auto alpha_casted = context.mark_node(make_shared<opset8::Convert>(context.get_input(2), context.get_input(0).get_element_type()));
            auto alpha_mul = context.mark_node(make_shared<opset8::Multiply>(context.get_input(0), alpha_casted));
            return { context.mark_node(make_shared<opset8::Subtract>(context.get_input(1), alpha_mul)) };
        }},

        { "aten::slice", [&]() -> OutputVector {
            ov::Output<ov::Node> dim = context.get_input(1);
            ov::Output<ov::Node> start = context.get_input(2);
            ov::Output<ov::Node> end = context.get_input(3);
            ov::Output<ov::Node> step = context.get_input(4);

            auto axis_0 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
            if (dim.get_partial_shape().rank().is_static() && dim.get_partial_shape().rank().get_length() == 0) {
                dim = context.mark_node(make_shared<opset8::Unsqueeze>(dim, axis_0));
            }
            if (start.get_partial_shape().rank().is_static() && start.get_partial_shape().rank().get_length() == 0) {
                start = context.mark_node(make_shared<opset8::Unsqueeze>(start, axis_0));
            }
            if (end.get_partial_shape().rank().is_static() && end.get_partial_shape().rank().get_length() == 0) {
                end = context.mark_node(make_shared<opset8::Unsqueeze>(end, axis_0));
            }
            if (step.get_partial_shape().rank().is_static() && step.get_partial_shape().rank().get_length() == 0) {
                step = context.mark_node(make_shared<opset8::Unsqueeze>(step, axis_0));
            }
            return { context.mark_node(make_shared<opset8::Slice>(context.get_input(0), start, end, step, dim)) };
        }},
        
        /* TODO: Don't know how to change it quickly to be compiled, consult with Maxim
        { "prim::ConstantChunk", [&]() -> OutputVector {
            auto chunks = node->i(attr::chunks); // FIXME: create actual attribute function
            auto dim = node->i(attr::dim);
            auto dim_const = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {dim}));
            auto split = context.mark_node(make_shared<opset8::Split>(context.get_input(0), dim_const, chunks));
            return split->outputs();
        }},
        */

        /* TODO: Don't need a special way to handle prim::ListConstruct as it doesn't provide any extra service extra to FW Node at this moment
        { "prim::ListConstruct", [&]() -> OutputVector {
            // TODO. Probably need to replace by a constant of size 0 in one of the following transformations that embed Lists to Tensors for OV.
        }},
        */

        /* Only makes sence if output type is not deducable, but it is not our case
        { "aten::__getitem__", [&]() -> OutputVector {
            // Should be handled in a special way: returned tensor is an alias for a part of input list
            // Temporary we replace this semantics loosing aliasing and just returning a part of input tensor.
            // Represent __getitem__ as generic node with only exception for returned type
            // We can infer output type base on the input type. Also we can verify that input type is really a list (TODO: is only list acceptable for __getitem__?)

            auto fw_node = make_shared<PtFrameworkNode>(decoder, context.inputs());
            OV_FRONTEND_REQUIRE(fw_node->outputs().size() == 1);
            // TODO: Deduce output type here; do we need it? Looks like PT has already derived all the types, just need to carefully recognize it in original graph
        }},
        */

        { "aten::append", [&]() -> OutputVector {
            // Returns a modified list but also modifies the original list as well. So it should replace original list entry point in tensor_map
            // with a new modified value. So returned value becomes a complete alise of input value with a list.
            // We replace the original entry point and produces new one for new entry point. It helps to maintain correct data dependencies and keep
            // graph connected.

            // We still using FW node to represent this op and going to call transformation that remakes it to supported sub graph

            auto fw_node = make_shared<PtFrameworkNode>(decoder, context.inputs());

            // Expect only a single output from aten::append
            OV_FRONTEND_REQUIRE(fw_node->outputs().size() == 1);

            // Next code is a hack to make alias for a value; in the final version it should be handled via something similar to AliasDB from PT
            auto new_alias = fw_node->output(0);
            auto input_list = decoder->input(0);
            OV_FRONTEND_REQUIRE(tensor_map.find(input_list) != tensor_map.end());   // for debug only, should be guaranteed
            tensor_map[input_list] = new_alias;  // replacing the original value which comes to append by the resulting list
            // TODO: this code won't work incorrectly if it is in a loop and list comes from outer scope
            return decoder->mark_node(fw_node)->outputs();
        }},

        { "prim::Constant", [&]() -> OutputVector {
            return context.as_constant();
        }}

    };

    auto it = converters.find(context.get_op_type());
    if(it != converters.end()) {
        //std::cout << "FOUND converter for " << context.get_op_type() << "\n";
        return it->second();
    } else {
        std::cout << "DIDN'T FIND converter for " << context.get_op_type() << "\n";
    }

    }
    // catch(pybind11::error_already_set& e) {
    //     std::cout << "Python exception: " << e << "\n";
    // }
    catch (std::runtime_error& e) {
        std::cout << std::flush;
        std::cout << "Exception happened during conversion: " << e.what() << " during conversion of node of type " << context.get_op_type() << '\n';
        std::cout << "Debug for node: " << std::endl;
        context.debug();
        std::cout << std::endl;
        std::cout << "End of debug output for node" << std::endl;
        //throw;
    }
    catch (...) {
        std::cout << "Some exception happened during convertion of node of type: " << context.get_op_type() << std::endl;
        std::cout << "Debug for node: " << std::endl;
        context.debug();
        std::cout << std::endl;
        std::cout << "End of debug output for node" << std::endl;
        //throw;
    }
    //if (node->kind() != prim::ListConstruct) {
    //    std::cout << "Making unsupported " << node->kind().toQualString() << std::endl;
    //    node->dump();
    //}
    
    // Create PtFrameworkNode for everything that wasn't able to be converted normally
    // Pay attention to subgraphs that may appear in the node
    //std::cerr << "[ DEBUG ] Before PtFramewokNode creation\n";
    auto fw_node = make_shared<PtFrameworkNode>(decoder, context.inputs());
    //std::cerr << "[ DEBUG ] After PtFramewokNode creation\n";

    for(size_t i = 0; i < decoder->get_subgraph_size(); ++i) {
        //std::cout << "Start converting subgraph\n";
        fw_node->add_subgraph(convert_pytorch_model(decoder->get_subgraph_decoder(i), tensor_map)); // tensor_map should contain both local context and all external contexts
        //std::cout << "End converting subgraph\n";
    }

    return decoder->mark_node(fw_node)->outputs();
}


std::shared_ptr<ov::Model> convert_pytorch_model(std::shared_ptr<Decoder> pytorch_model, const TensorMap& external_tensor_map) {
    std::shared_ptr<ov::Model> resulting_model; // define here to make a conversion in a nested scope
    {
    ParameterVector parameters;

    TensorMap tensor_map;   // tensor map of the current context

    //std::cerr << "+++++before++++\n";
    // Go over all pytorch_model inputs and register them in the tensor map:
    auto inputs = pytorch_model->inputs();
    //std::cerr << "+++++after++++++\n";
    std::cout << "[  ---  DEBUG --- ] convert_pytorch_model: number of inputs: " << inputs.size() << '\n';
    for (int i = 0; i < inputs.size(); ++i) {
        std::cout << "Input: " << i << ": " << inputs[i] << "\n";
        PartialShape ps = pytorch_model->get_input_shape(i);
        std::cout << "PartialShape = " << ps << "\n";
        Any type = pytorch_model->get_input_type(i);
        std::cout << "Custom Type = " << type.type_info().name() << std::endl;
        auto parameter = std::make_shared<opset7::Parameter>(ov::element::custom, type, ps);
        std::cout << "Parameter: " << parameter << std::endl;
        parameters.push_back(parameter);
        auto order = pytorch_model->get_input_transpose_order(i);
        if (order.size() > 0 && !std::is_sorted(order.begin(), order.end())) {
            OV_FRONTEND_REQUIRE(ps.is_static());    // TODO: make dynamic
            auto sh = ps.get_shape();
            Shape new_shape(sh.size());
            for (int i = 0; i < sh.size(); i++) {
                new_shape[order[i]] = sh[i];
            }
            auto shape_const = opset7::Constant::create(element::i64, { new_shape.size() }, new_shape);
            auto reshape = std::make_shared<opset7::Reshape>(parameter, shape_const, false);
            auto order_const = opset7::Constant::create(element::i32, { order.size() }, order);
            auto transpose = std::make_shared<opset7::Transpose>(reshape, order_const);
            tensor_map[pytorch_model->input(i)] = transpose;
        } else {
            tensor_map[pytorch_model->input(i)] = parameter;
        }
    }

    
    auto node_visitor = [&](std::shared_ptr<Decoder> node)
    {
        //std::cerr << "Node convert start" << std::endl;

        // Explore all inputs of node. Node may refer to input value that hasn't been created in the current scope.
        // But this value can be found in the outer scope, for this purpose we need to search node in external_tensor_map as well

        auto raw_inputs = node->inputs();
        for(size_t i = 0; i < raw_inputs.size(); ++i) {
            auto input = node->input(i);
            if(tensor_map.find(input) == tensor_map.end()) {
                //std::cout << "Trampoline for input index " << i << " with value " << input << "\n";
                // input refers value in the outer scope, need to create a new Parameter in the current scope
                // TODO: Connect outer scope and inner scope properly -- should be handled at the level of that operation that introduced this nest of scopes (e.g. loop or if)
                // TODO: Eliminate duplication with the main code for Parameters creation
                // TODO: There is no real search for values in outer scope because we don't need to link the usage and definition together at this point -- need to do that otherwise graph will fall apart
                PartialShape ps = node->get_input_shape(i);
                auto parameter = std::make_shared<opset7::Parameter>(node->get_input_type(i), ps);
                parameters.push_back(parameter);
                // TODO: Missing get_input_transpose_order handling for not trivial layouts
                tensor_map[input] = parameter;
                //std::cout << "Parameter created\n";
            }
        }

        auto converted_outputs = convert_node(node, tensor_map);
        //std::cerr << "Node convert before outputs" << std::endl;

        auto fw_outputs = node->outputs();

        // TODO: Make sure that mapping of fw_outputs to converted_outputs does always work
        // FIXME: Now it is not true for at least prim::Constant
        for(size_t i = 0; i < converted_outputs.size(); ++i) {
            size_t fw_tensor_id = node->output(i);
            if(tensor_map.find(fw_tensor_id) != tensor_map.end()) {
                //std::cerr << "Duplicated producer for tensor with id = " << fw_tensor_id << " discovered at output "
                //    << "port " << i << " of node " << node->kind().toQualString() << "\n";
                throw std::runtime_error("Duplicated producer for PT value with unique ID: " + std::to_string(fw_tensor_id));
            }

            // Output shape of converted node should match the original output shape
            //std::cerr << "[ DEBUG ] PT output shape = " << get_ov_shape(fw_outputs[i]) << '\n';
            //std::cerr << "[ DEBUG ] OV output shape = " << converted_outputs[i].get_partial_shape() << '\n';
            //OV_FRONTEND_REQUIRE(get_ov_shape(fw_outputs[i]) == converted_outputs[i].get_partial_shape());

            tensor_map[fw_tensor_id] = converted_outputs[i];
        }
        //std::cout << "Node convert end" << std::endl;
    };

    OV_FRONTEND_REQUIRE(pytorch_model->get_subgraph_size() == 1);
    pytorch_model->visit_subgraph(0, node_visitor);
    //std::cout << "All nodes convert end" << std::endl;

    ResultVector results;
    //std::cerr << "Outputs:\n";
    for (size_t i = 0; i < pytorch_model->num_of_outputs(); ++i) {
        //std::cerr << i << "\n";
        size_t id = pytorch_model->output(i);
        //std::cout << "value = " << id << '\n';
        //std::cout << "X\n";
        if(tensor_map.find(id) == tensor_map.end()) {
            // Not found in this scope, searching in the outer scope
            // TODO: do real search here, skipped for now

            auto parameter = std::make_shared<opset7::Parameter>(element::dynamic, PartialShape::dynamic());
            parameters.push_back(parameter);
            tensor_map[id] = parameter;
            //std::cout << "Added new parameter based on external value\n";
        }
        auto ov_output = tensor_map[id];
        //std::cout << "X\n";
        auto order = pytorch_model->get_output_transpose_order(i);
        //std::cout << "X\n";
        if (order.size() > 0 && !std::is_sorted(order.begin(), order.end())) {
            throw "Output strides have wrong order.";
        }
        //std::cout << "X\n";
        //std::cout << ov_output << '\n';
        auto result = std::make_shared<opset7::Result>(ov_output);
        //std::cout << "X\n";
        results.push_back(result);
        //std::cerr << "Cluster result " << value->unique() << " with shape " << result->get_output_partial_shape(0) << "\n";
    }
        //std::cout << "Y\n";

    for(size_t i = 0; i < parameters.size(); ++i) {
        auto parameter = parameters[i];
        //std::cerr << "parameter[" << i << "].shape = "
        //    << parameter->get_output_shape(0) << ", consumers: " << parameter->output(0).get_target_inputs().size() << "\n";
    }
    //std::cout << "Convert end" << std::endl;
    //std::cout << "Number of values collected: " << tensor_map.size() << "\n";

    resulting_model = std::make_shared<ov::Model>(results, parameters);
    
    // Did a conversion in a nested scope to automatically remove any holders of nodes except those in the graph
    }

    // TODO: Propose better solution for the next code block
    // Usually if nn.Module.forward is given as a source model for conversion, there is the first Parameter
    // that represents original `self` argument in forward(self, ...). `self` shouldn't play any role in model inference
    // if model is completelly frozed and all methods are inlined. So we check if it doesn't have any consumers
    // in the finally converted model and remove this parameter. This parameter should have index 0.
    if(resulting_model->get_parameters().size() > 0) {
        auto self = resulting_model->get_parameters()[0];
        if(self->output(0).get_target_inputs().empty()) {
            // There is no consumers: safe to remove
            std::cout << "[ WARNING ] Removing parameter[0] in converted Pytorch model, because it is never used and treated as `self`\n";
            resulting_model->remove_parameter(self);
        }
    }    
    
    return resulting_model;
}

std::shared_ptr<Model> FrontEnd::convert(const ov::frontend::InputModel::Ptr& model) const {
    try {
        //std::cerr << "[   HERE   ]\n";
        auto pytorch_model = std::dynamic_pointer_cast<pytorch::InputModel>(model);
        // TODO: Remove this super-hack, tensor_map should be local for each conversion activity, see more info where tensor_map is defined now
        TensorMap external_tensor_map; // intentionally empty because there is no external context
        auto model = convert_pytorch_model(pytorch_model->m_model, external_tensor_map);
        apply_pytorch_conversion_transforms(model);
        return model;
    } catch (const std::runtime_error& e) {
        std::cerr << "[ ERROR ] Error while converting pytorch model: " << e.what() << "\n";
        std::cerr << "Rethrowing. Misleading error message from pybind11 may come next. TODO.";
        throw;
    }
}

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    //std::cout << "[  ----- DEBUG ------ ] supported_impl with " << variants.size() << " arguments\n";
    return false;
}

ov::frontend::InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    //std::cout << "[  ----- DEBUG -----  ] load_impl with " << variants.size() << " parameters\n";
    if(variants.size() != 1) {
        throw std::runtime_error("Pytorch frontend supports exactly one parameter in model representation, got " +
        std::to_string(variants.size()) + "instead.");
    }
    auto decoder = variants[0].as<std::shared_ptr<Decoder>>();
    //std::cout << "Recognized decoder: " << decoder << "\n";
    return std::make_shared<pytorch::InputModel>(decoder);
}

}
}
}
