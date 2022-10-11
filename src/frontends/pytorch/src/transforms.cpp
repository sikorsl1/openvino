#include "transforms.hpp"

#include <openvino/frontend/pytorch/decoder.hpp>
#include <openvino/op/util/framework_node.hpp>
#include <openvino/opsets/opset9.hpp>
#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <tuple>

#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

using ov::op::util::FrameworkNode;
using ov::pass::MatcherPass;
using ov::pass::pattern::any_input;
using ov::pass::pattern::Matcher;
using ov::pass::pattern::wrap_type;
using std::make_shared;
using std::shared_ptr;

std::tuple<bool, Any> is_list_of_tensors(const descriptor::Tensor& tensor) {
    if (tensor.get_element_type() != element::custom)
        return std::make_tuple(false, Any());
    Any custom_type = tensor.get_custom_element_type();
    if (custom_type.empty()) {
        return std::make_tuple(false, Any());
    }
    //std::cerr << tensor.get_names() << "\n";
    std::cerr << "Custom type: ";
    std::cerr << custom_type.type_info().name();
    std::cerr << " after\n";

    // TODO: Sometimes there is uninitialized Any in custom_type, need to understand where it comes from

    if (!custom_type.is<Type::List>()) {
        //std::cerr << "[ PASS INFO ] Skipping FW Node because it produces not a list\n";
        return std::make_tuple(false, custom_type);
    }

    Any element_type = custom_type.as<Type::List>().element_type;

    if (!element_type.is<Type::Tensor>()) {
        //std::cerr << "[ PASS INFO ] Skipping FW Node because element of a List is not a Tensor\n";
        return std::make_tuple(false, custom_type);
    }

    return std::make_tuple(true, custom_type);
}

std::shared_ptr<FrameworkNode> make_list_pack(const OutputVector& inputs, Any output_type, const PartialShape& shape) {
    auto list_pack = make_shared<FrameworkNode>(inputs, 1);  // 6 inputs -- 1 output
    if(output_type.empty()) {
        throw std::runtime_error("Attemt to call make_list_pack with empty output_type");
    }
    list_pack->set_custom_output_type(0, output_type, shape);
    op::util::FrameworkNodeAttrs attrs;
    attrs.set_type_name("PTFE::ListPack");
    list_pack->set_attrs(attrs);
    list_pack->validate_and_infer_types();
    return list_pack;
}

std::shared_ptr<FrameworkNode> cast_internal_node(std::shared_ptr<Node> node, const std::string& type) {
    auto fw_node = std::dynamic_pointer_cast<FrameworkNode>(node);
    if (!fw_node) {
        //std::cerr << "[ ERROR ] Incorrect matcher triggering\n";
        return nullptr;
    }
    if (fw_node->get_attrs().find("PtTypeName") != fw_node->get_attrs().end()) {
        // This is FW node, not PT FW internal node, don't mix them
        //std::cerr << "[ ERROR ] This is PtTypeName-node: " << fw_node->get_attrs().at("PtTypeName") << "\n";
        return nullptr;
    }
    if (fw_node->get_attrs().get_type_name() != type) {
        //std::cerr << "[ ERROR ] Not expected type: " << fw_node->get_type_name() << "\n";
        return nullptr;
    }

    return fw_node;
}

class ListConstructPass : public MatcherPass {
public:
    OPENVINO_RTTI("PytorchFrontendListConstructPass", "0");
    ListConstructPass() {
        auto convert = wrap_type<FrameworkNode>();

        ov::matcher_pass_callback callback = [](Matcher& m) {
            auto node = cast_fw_node(m.get_match_root(), "prim::ListConstruct");
            if (!node)
                return false;
            std::cerr << node << "\n";
            const descriptor::Tensor& list_output = node->output(0).get_tensor();

            auto custom_types = is_list_of_tensors(list_output);

            if (!std::get<0>(custom_types)) {
                return false;
            }

            auto custom_type = std::get<1>(custom_types);
            if(custom_type.empty()) {
                throw std::runtime_error("Custom element type is empty");
            }

            //std::cerr << "[ PASS INFO ] Start transformation\n";

            // Replace a single ListConstruct with 6 constant tensors:
            //   - beginnings of tensor elements of type i32 and shape [0]
            //   - endings of tensor elements of type i32 and shape [0]
            //   - beginnnigs of shape dimensions of type i32 and shape [0]
            //   - endings of tensor elements of type i32 and shape [0]
            //   - shape dimensions of type i32 and shape [0]
            //   - tensor elements flattened of type i32 (any type) and shape [0]
            // Type of elements for the latest tensor is not really known at the moment
            // Even worse, it can be dynamic and differ among elements
            // So for now we are selecting any type, say f32

            // Make one i32 constant for all 6 inputs
            auto empty_const = opset9::Constant::create(element::i32, {0}, {});
            OutputVector inputs(6, empty_const);

            auto list_pack = make_list_pack(inputs, custom_type, node->get_output_partial_shape(0));
            replace_node(node, list_pack);

            return true;
        };

        auto m = make_shared<Matcher>(convert, "PytorchFrontendListConstructPass");
        this->register_matcher(m, callback);
    }
};

class DecomposeListParameters : public pass::ModelPass {
public:
    bool run_on_model(const std::shared_ptr<Model>& model) override {
        // Search for Parameter with List[Tensor] types

        ParameterVector parameters = model->get_parameters();
        ParameterVector new_parameters;  // collect decomposed parameters
        for (size_t i = 0; i < parameters.size(); ++i) {
            auto parameter = parameters[i];
            //std::cerr << "[ PARAMETER ] " << i << "\n";
            std::cerr << parameter << "\n";

            auto custom_types = is_list_of_tensors(parameter->get_output_tensor(0));

            if (std::get<0>(custom_types)) {
                // Decompose each parameter that represetns the list of tensors to 6 inputs
                // Element type of the parameter that represents tensor elements is unknown (leave it dynamic)
                // Keep original parameters in the model, just detach it from the network -- to avoid parameters
                // renumbering for unchanged parameters
                // TODO: Reorganize parameters handling (second level of parameters interpretation)

                OutputVector inputs_for_list_pack;

                // for tensors offsets and shapes
                for (size_t i = 0; i < 5; ++i) {
                    auto new_parameter =
                        make_shared<opset9::Parameter>(element::i32, PartialShape{Dimension::dynamic()});
                    new_parameters.push_back(new_parameter);
                    inputs_for_list_pack.push_back(new_parameter);
                    // TODO: add links via RT info between original parameter and new ones
                }

                // for tensor elements
                auto new_parameter =
                    make_shared<opset9::Parameter>(element::dynamic, PartialShape{Dimension::dynamic()});
                new_parameters.push_back(new_parameter);
                inputs_for_list_pack.push_back(new_parameter);

                auto list_pack = make_list_pack(inputs_for_list_pack,
                                                std::get<1>(custom_types),
                                                parameter->get_output_partial_shape(0));
                replace_node(parameter, list_pack);

                model->remove_parameter({parameter});
            }
        }

        model->add_parameters(new_parameters);

        return true;
    }
};

class DecomposeGetItem : public MatcherPass {
public:
    OPENVINO_RTTI("PytorchFrontendDecomposeGetItem", "0");
    DecomposeGetItem() {
        auto begins = any_input();
        auto ends = any_input();
        auto shape_begins = any_input();
        auto shape_ends = any_input();
        auto shape_dims = any_input();
        auto tensor_elements = any_input();
        auto list_pack =
            wrap_type<FrameworkNode>({begins, ends, shape_begins, shape_ends, shape_dims, tensor_elements});
        auto index = any_input();
        auto get_item = wrap_type<FrameworkNode>({list_pack, index});

        ov::matcher_pass_callback callback = [=](Matcher& m) {
            auto matches = m.get_pattern_map();

            auto get_item_node = cast_fw_node(matches.at(get_item), "aten::__getitem__");
            if (!get_item_node)
                return false;

            auto list_pack_node = cast_internal_node(matches.at(list_pack), "PTFE::ListPack");
            if (!list_pack_node)
                return false;

            auto zero = opset9::Constant::create(element::i32, {1}, {0});
            auto one = opset9::Constant::create(element::i32, {1}, {1});
            auto mask = std::vector<int64_t>{0};

            // Prepare index to be 1D tensor to have predictable ranks after Gather for StridedSlice
            auto index_1D = make_shared<opset9::Reshape>(matches.at(index), one, false);

            // Slice out region with elements relevant to required item from tensor_elements based on begins and ends
            auto elements =
                make_shared<opset9::StridedSlice>(matches.at(tensor_elements),
                                                  make_shared<opset9::Gather>(matches.at(begins), index_1D, zero),
                                                  make_shared<opset9::Gather>(matches.at(ends), index_1D, zero),
                                                  mask,
                                                  mask);

            // Get region of shape dimensions that belongs to the selected item
            auto shape =
                make_shared<opset9::StridedSlice>(matches.at(shape_dims),
                                                  make_shared<opset9::Gather>(matches.at(shape_begins), index_1D, zero),
                                                  make_shared<opset9::Gather>(matches.at(shape_ends), index_1D, zero),
                                                  mask,
                                                  mask);

            // Reshape elements to have a given shape -- this is our result
            auto item = make_shared<opset9::Reshape>(elements, shape, false);

            replace_node(get_item_node, item);

            return true;
        };

        auto m = make_shared<Matcher>(get_item, "PytorchFrontendDecomposeGetItem");
        this->register_matcher(m, callback);
    }
};

class DecomposeAppend : public MatcherPass {
public:
    OPENVINO_RTTI("PytorchFrontendDecomposeAppend", "0");
    DecomposeAppend() {
        auto begins = any_input();
        auto ends = any_input();
        auto shape_begins = any_input();
        auto shape_ends = any_input();
        auto shape_dims = any_input();
        auto elements = any_input();
        auto list_pack = wrap_type<FrameworkNode>({begins, ends, shape_begins, shape_ends, shape_dims, elements});
        auto item = any_input();
        auto append = wrap_type<FrameworkNode>({list_pack, item});

        ov::matcher_pass_callback callback = [=](Matcher& m) {
            // TODO: replace by values whenever possible
            auto matches = m.get_pattern_map();

            auto append_node = cast_fw_node(matches.at(append), "aten::append");
            if (!append_node)
                return false;

            auto list_pack_node = cast_internal_node(matches.at(list_pack), "PTFE::ListPack");
            if (!list_pack_node)
                return false;

            std::cerr << append_node << "\n";
            auto custom_types = is_list_of_tensors(append_node->get_output_tensor(0));

            if (!std::get<0>(custom_types)) {
                return false;
            }

            auto custom_type = std::get<1>(custom_types);

            // Appending new shape dimensions and producing adjusted versions of shape_begins and shape_ends
            auto shape = make_shared<opset9::ShapeOf>(matches.at(item), element::i32);
            auto cur_shape_dims_size = make_shared<opset9::ShapeOf>(matches.at(shape_dims), element::i32);
            auto new_shape_begins =
                make_shared<opset9::Concat>(NodeVector{matches.at(shape_begins), cur_shape_dims_size}, 0);
            auto new_shape_dims = make_shared<opset9::Concat>(NodeVector{matches.at(shape_dims), shape}, 0);
            auto new_shape_dims_size = make_shared<opset9::ShapeOf>(new_shape_dims, element::i32);
            auto new_shape_ends =
                make_shared<opset9::Concat>(NodeVector{matches.at(shape_ends), new_shape_dims_size}, 0);

            // Appending new elements after flattening to existing elements

            auto item_flatten = make_shared<opset9::Reshape>(matches.at(item),
                                                             opset9::Constant::create(element::i32, {1}, {-1}),
                                                             false);
            auto new_begins = make_shared<opset9::Concat>(
                NodeVector{matches.at(begins), make_shared<opset9::ShapeOf>(matches.at(elements), element::i32)},
                0);

            auto initial_elements_const = std::dynamic_pointer_cast<opset9::Constant>(matches.at(elements));
            // New elements content depends on whether we appending to an empty list or not
            //std::cerr << "Tried to detect constant here: " << matches.at(elements) << "\n";
            //std::cerr << "Is const: " << bool(initial_elements_const) << " )))))))))))))))))\n";
            //if (initial_elements_const) {
            //    std::cerr << "Shape: " << initial_elements_const->get_output_partial_shape(0) << "\n";
            //    std::cerr << "Size: " << shape_size(initial_elements_const->get_output_shape(0))
            //              << "  )))))))))))))))))\n";
            //}
            auto new_elements =
                (initial_elements_const && shape_size(initial_elements_const->get_output_shape(0)) == 0)
                    ? shared_ptr<Node>(item_flatten)
                    :  // empty initial list -- just take appended elements as a new content for the list; derive type
                       // from that tensor
                    shared_ptr<Node>(make_shared<opset9::Concat>(NodeVector{matches.at(elements), item_flatten},
                                                                 0));  // existing list, just concat

            auto new_ends = make_shared<opset9::Concat>(
                NodeVector{matches.at(ends), make_shared<opset9::ShapeOf>(new_elements, element::i32)},
                0);

            auto new_list_pack =
                make_list_pack({new_begins, new_ends, new_shape_begins, new_shape_ends, new_shape_dims, new_elements},
                               std::get<1>(custom_types),
                               append_node->get_output_partial_shape(0));

            replace_node(append_node, new_list_pack);

            return true;
        };

        auto m = make_shared<Matcher>(append, "PytorchFrontendDecomposeAppend");
        this->register_matcher(m, callback);
    }
};

class DecomposeListResults : public pass::ModelPass {
public:
    bool run_on_model(const std::shared_ptr<Model>& model) override {
        // Search for Parameter with List[Tensor] types

        bool at_least_one_decomposed = false;

        ResultVector results =
            model->get_results();  // make a copy, leter results in the model are going to be modified

        for (size_t i = 0; i < results.size(); ++i) {
            auto result = results[i];
            std::cerr << result << "\n";
            auto custom_types = is_list_of_tensors(result->get_input_tensor(0));

            auto list_pack = cast_internal_node(result->get_input_node_shared_ptr(0), "PTFE::ListPack");

            if (std::get<0>(custom_types) && list_pack) {
                // Replace a single result with 6 results, per each input of parent list_pack

                auto inputs = list_pack->inputs();
                for (auto input : inputs) {
                    model->add_results({make_shared<opset9::Result>(input.get_source_output())});
                    // TODO: Keep tracking between original and new Results
                }

                model->remove_result(result);
                at_least_one_decomposed = true;
            }
        }

        return at_least_one_decomposed;
    }
};

void apply_pytorch_conversion_transforms(std::shared_ptr<ov::Model> model) {
    //return;
    std::cerr << "[ PASS INFO ] apply_pytorch_conversion_transforms\n";
    pass::Manager manager;
    manager.register_pass<DecomposeListParameters>();

    auto matchers = manager.register_pass<pass::GraphRewrite>();
    matchers->add_matcher<ListConstructPass>();
    matchers->add_matcher<DecomposeGetItem>();
    matchers->add_matcher<DecomposeAppend>();

    manager.register_pass<pass::Validate>();
    manager.register_pass<DecomposeListResults>();

    manager.run_passes(model);
    std::cerr << "[ PASS INFO ] Passes passed\n";
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov