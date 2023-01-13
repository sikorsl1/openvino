// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <openvino/frontend/manager.hpp>
#include <openvino/opsets/opset9.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/transpose_sinking_binary.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"

namespace {

using NodePtr = std::shared_ptr<ov::Node>;
using ModelPtr = std::shared_ptr<ov::Model>;
using Output = ov::Output<ov::Node>;

namespace {
std::string to_string(const ov::Shape& shape) {
    std::ostringstream result;
    result << "{";
    for (size_t idx = 0; idx < shape.size(); ++idx) {
        if (idx)
            result << ",";
        result << shape[idx];
    }
    result << "}";
    return result.str();
}
}  // namespace

// ----------------------------------------------------------------------------

class IBinaryFactory {
public:
    IBinaryFactory(const std::string& type_name) : type_name_(type_name) {}
    virtual ~IBinaryFactory() = default;
    virtual NodePtr create(NodePtr parent_left_node, NodePtr parent_right_node) const = 0;
    const std::string& getTypeName() const {
        return type_name_;
    }

private:
    const std::string type_name_;
};

using BinaryFactoryPtr = std::shared_ptr<IBinaryFactory>;

template <typename BinaryT>
class BinaryFactory : public IBinaryFactory {
public:
    BinaryFactory(const std::string& type_name) : IBinaryFactory(type_name) {}
    NodePtr create(NodePtr parent_left_node, NodePtr parent_right_node) const override {
        return std::make_shared<BinaryT>(parent_left_node, parent_right_node);
    }
};

template <typename BinaryT>
BinaryFactoryPtr CreateBinaryFactory(const std::string& type_name) {
    return std::make_shared<BinaryFactory<BinaryT>>(type_name);
}

// ----------------------------------------------------------------------------

class IPassFactory {
public:
    IPassFactory(const std::string& type_name) : type_name_(type_name) {}
    virtual ~IPassFactory() = default;
    virtual void registerPass(ov::pass::Manager& pass_manager) const = 0;
    const std::string& getTypeName() const {
        return type_name_;
    }

private:
    const std::string type_name_;
};

using PassFactoryPtr = std::shared_ptr<IPassFactory>;

template <typename PassT>
class PassFactory : public IPassFactory {
public:
    PassFactory(const std::string& type_name) : IPassFactory(type_name) {}
    void registerPass(ov::pass::Manager& pass_manager) const override {
        pass_manager.register_pass<PassT>();
    }
};

#define CREATE_PASS_FACTORY(pass_name) std::make_shared<PassFactory<ov::pass::pass_name>>(#pass_name)

#undef CREATE_BINARY_FACTORY
#define CREATE_BINARY_FACTORY(type_name) CreateBinaryFactory<ov::opset9::type_name>(#type_name)
std::vector<BinaryFactoryPtr> binary_factories = {CREATE_BINARY_FACTORY(Add),
                                                  CREATE_BINARY_FACTORY(Divide),
                                                  CREATE_BINARY_FACTORY(Maximum),
                                                  CREATE_BINARY_FACTORY(Minimum),
                                                  CREATE_BINARY_FACTORY(Mod),
                                                  CREATE_BINARY_FACTORY(Multiply),
                                                  CREATE_BINARY_FACTORY(Power),
                                                  CREATE_BINARY_FACTORY(SquaredDifference),
                                                  CREATE_BINARY_FACTORY(Subtract)};
#undef CREATE_BINARY_FACTORY

std::vector<size_t> binary_operations_numbers = {1, 10};

std::vector<size_t> binary_transpose_input_indexes = {0, 1};

}  // namespace

namespace binary {
namespace single_consumer {
namespace forward {
namespace one_input_transpose {

std::shared_ptr<ov::Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                          size_t num_binary_ops,
                                          ov::element::Type input_type,
                                          size_t binary_transpose_input_idx) {
    const ov::Shape input_shape{1, 96, 55, 55};
    const ov::Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, const_shape, ov::Shape{1});
        if (!binary_transpose_input_idx)
            in_op = binary_factory->create(in_op, in_constant);
        else
            in_op = binary_factory->create(in_constant, in_op);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{in_op}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                                   size_t num_binary_ops,
                                                   ov::element::Type input_type,
                                                   size_t binary_transpose_input_idx) {
    const ov::Shape input_shape{1, 96, 55, 55};
    const ov::Shape const_shape{1, 55, 55, 96};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, const_shape, ov::Shape{1});

        auto transpose_reversed_const =
            std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
        auto transpose_reversed = std::make_shared<ov::opset9::Transpose>(in_constant, transpose_reversed_const);

        if (!binary_transpose_input_idx)
            in_op = binary_factory->create(in_op, transpose_reversed);
        else
            in_op = binary_factory->create(transpose_reversed, in_op);
    }

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

    return std::make_shared<ov::Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

}  // namespace one_input_transpose

namespace double_transpose {
std::shared_ptr<ov::Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                          size_t num_binary_ops,
                                          ov::element::Type input_type) {
    const ov::Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});
        auto ng_order1 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose1 = std::make_shared<ov::opset9::Transpose>(in_constant, ng_order1);

        in_op = binary_factory->create(in_op, transpose1);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{in_op}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                                   size_t num_binary_ops,
                                                   ov::element::Type input_type) {
    const ov::Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});

        auto ng_order1 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose1 = std::make_shared<ov::opset9::Transpose>(in_constant, ng_order1);

        auto transpose_reversed_const =
            std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
        auto transpose_reversed = std::make_shared<ov::opset9::Transpose>(transpose1, transpose_reversed_const);

        in_op = binary_factory->create(in_op, transpose_reversed);
    }

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

    return std::make_shared<ov::Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

}  // namespace double_transpose
}  // namespace forward

namespace backward {
namespace one_input_transpose {
std::shared_ptr<ov::Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                          size_t num_binary_ops,
                                          ov::element::Type input_type,
                                          size_t binary_transpose_input_idx) {
    const ov::Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    NodePtr in_op = X;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});
        if (!binary_transpose_input_idx)
            in_op = binary_factory->create(in_op, in_constant);
        else
            in_op = binary_factory->create(in_constant, in_op);
    }

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(in_op, ng_order0);

    return std::make_shared<ov::Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                                   size_t num_binary_ops,
                                                   ov::element::Type input_type,
                                                   size_t binary_transpose_input_idx) {
    const ov::Shape input_shape{1, 96, 55, 55};

    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

    NodePtr in_op = transpose0;
    for (size_t i = 0; i < num_binary_ops; ++i) {
        auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, input_shape, ov::Shape{1});

        auto ng_order = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
        auto transpose = std::make_shared<ov::opset9::Transpose>(in_constant, ng_order);

        if (!binary_transpose_input_idx)
            in_op = binary_factory->create(in_op, transpose);
        else
            in_op = binary_factory->create(transpose, in_op);
    }

    return std::make_shared<ov::Model>(ov::OutputVector{in_op}, ov::ParameterVector{X});
}
}  // namespace one_input_transpose
}  // namespace backward
}  // namespace single_consumer
}  // namespace binary

using CreateGraphBinaryF = std::function<std::shared_ptr<ov::Model>(BinaryFactoryPtr unary_factory,
                                                                    size_t num_binary_ops,
                                                                    ov::element::Type input_type,
                                                                    size_t binary_transpose_input_idx)>;

using TestBinaryParams = std::tuple<BinaryFactoryPtr,
                                    PassFactoryPtr,
                                    size_t,             /* num_binary_ops */
                                    CreateGraphBinaryF, /* model_factory */
                                    CreateGraphBinaryF, /* reference_model_factory */
                                    ov::element::Type,  /* input type */
                                    size_t>;            /* binary_transpose_input_idx */

class TransposeSinkingBinaryTestFixture : public ::testing::WithParamInterface<TestBinaryParams>,
                                          public TransformationTestsF {
public:
    static std::string get_test_name(const testing::TestParamInfo<TestBinaryParams>& obj) {
        BinaryFactoryPtr binary_factory;
        PassFactoryPtr pass_factory;
        size_t num_binary_ops;
        CreateGraphBinaryF model_factory;
        CreateGraphBinaryF reference_model_factory;
        ov::element::Type input_type;
        size_t binary_transpose_input_idx;
        std::tie(binary_factory,
                 pass_factory,
                 num_binary_ops,
                 model_factory,
                 reference_model_factory,
                 input_type,
                 binary_transpose_input_idx) = obj.param;

        std::ostringstream test_name;
        test_name << "binary_factory=" << binary_factory->getTypeName() << "_";
        test_name << "pass_factory=" << pass_factory->getTypeName() << "_";
        test_name << "num_binary_ops=" << num_binary_ops << "_";
        test_name << "input_type=" << input_type << "_";
        test_name << "binary_transpose_input_idx=" << binary_transpose_input_idx;

        return test_name.str();
    }
};

TEST_P(TransposeSinkingBinaryTestFixture, CompareFunctions) {
    BinaryFactoryPtr binary_factory;
    PassFactoryPtr pass_factory;
    size_t num_binary_ops;
    CreateGraphBinaryF model_factory;
    CreateGraphBinaryF reference_model_factory;
    ov::element::Type input_type;
    size_t binary_transpose_input_idx;
    std::tie(binary_factory,
             pass_factory,
             num_binary_ops,
             model_factory,
             reference_model_factory,
             input_type,
             binary_transpose_input_idx) = this->GetParam();

    model = model_factory(binary_factory, num_binary_ops, input_type, binary_transpose_input_idx);
    model_ref = reference_model_factory(binary_factory, num_binary_ops, input_type, binary_transpose_input_idx);
    pass_factory->registerPass(manager);
}

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingBinaryForwardTestSuite,
    TransposeSinkingBinaryTestFixture,
    ::testing::Combine(
        ::testing::ValuesIn(binary_factories),
        ::testing::Values(CREATE_PASS_FACTORY(TransposeSinkingBinaryElementwiseForward)),
        ::testing::ValuesIn(binary_operations_numbers),
        ::testing::Values(binary::single_consumer::forward::one_input_transpose::CreateFunction),
        ::testing::Values(binary::single_consumer::forward::one_input_transpose::CreateReferenceFunction),
        ::testing::Values(ov::element::f32),
        ::testing::ValuesIn(binary_transpose_input_indexes)),
    TransposeSinkingBinaryTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingBinaryBackwardTestSuite,
    TransposeSinkingBinaryTestFixture,
    ::testing::Combine(
        ::testing::ValuesIn(binary_factories),
        ::testing::Values(CREATE_PASS_FACTORY(TransposeSinkingBinaryElementwiseBackward)),
        ::testing::ValuesIn(binary_operations_numbers),
        ::testing::Values(binary::single_consumer::backward::one_input_transpose::CreateFunction),
        ::testing::Values(binary::single_consumer::backward::one_input_transpose::CreateReferenceFunction),
        ::testing::Values(ov::element::f32),
        ::testing::ValuesIn(binary_transpose_input_indexes)),
    TransposeSinkingBinaryTestFixture::get_test_name);

// --------------------------------------------------------------------------------------

using CreateGraphBinaryIncompatShapesF = std::function<std::shared_ptr<ov::Model>(BinaryFactoryPtr unary_factory,
                                                                                  ov::element::Type input_type,
                                                                                  ov::Shape input_shape,
                                                                                  ov::Shape constant_shape,
                                                                                  size_t binary_transpose_input_idx)>;

using TestBinaryIncompatShapesParams = std::tuple<BinaryFactoryPtr,
                                                  PassFactoryPtr,
                                                  ov::Shape,                        /* input shape */
                                                  ov::Shape,                        /* constant_shape */
                                                  CreateGraphBinaryIncompatShapesF, /* model_factory */
                                                  CreateGraphBinaryIncompatShapesF, /* reference_model_factory */
                                                  ov::element::Type,                /* input type */
                                                  size_t>;                          /* binary_transpose_input_idx */

class TransposeSinkingBinaryIncompatShapesTestFixture
    : public ::testing::WithParamInterface<TestBinaryIncompatShapesParams>,
      public TransformationTestsF {
public:
    static std::string get_test_name(const testing::TestParamInfo<TestBinaryIncompatShapesParams>& obj) {
        BinaryFactoryPtr binary_factory;
        PassFactoryPtr pass_factory;
        ov::Shape input_shape;
        ov::Shape constant_shape;
        CreateGraphBinaryIncompatShapesF model_factory;
        CreateGraphBinaryIncompatShapesF reference_model_factory;
        ov::element::Type input_type;
        size_t binary_transpose_input_idx;
        std::tie(binary_factory,
                 pass_factory,
                 input_shape,
                 constant_shape,
                 model_factory,
                 reference_model_factory,
                 input_type,
                 binary_transpose_input_idx) = obj.param;

        std::ostringstream test_name;
        test_name << "binary_factory=" << binary_factory->getTypeName() << "_";
        test_name << "pass_factory=" << pass_factory->getTypeName() << "_";
        test_name << "input_shape=" << to_string(input_shape) << "_";
        test_name << "constant_shape=" << to_string(constant_shape) << "_";
        test_name << "input_type=" << input_type << "_";
        test_name << "binary_transpose_input_idx=" << binary_transpose_input_idx;

        return test_name.str();
    }
};

TEST_P(TransposeSinkingBinaryIncompatShapesTestFixture, CompareFunctions) {
    BinaryFactoryPtr binary_factory;
    PassFactoryPtr pass_factory;
    ov::Shape input_shape;
    ov::Shape constant_shape;
    CreateGraphBinaryIncompatShapesF model_factory;
    CreateGraphBinaryIncompatShapesF reference_model_factory;
    ov::element::Type input_type;
    size_t binary_transpose_input_idx;
    std::tie(binary_factory,
             pass_factory,
             input_shape,
             constant_shape,
             model_factory,
             reference_model_factory,
             input_type,
             binary_transpose_input_idx) = this->GetParam();

    model = model_factory(binary_factory, input_type, input_shape, constant_shape, binary_transpose_input_idx);
    model_ref =
        reference_model_factory(binary_factory, input_type, input_shape, constant_shape, binary_transpose_input_idx);
    pass_factory->registerPass(manager);
}

namespace binary {
namespace single_consumer {
namespace backward {
namespace incompat_shapes {

std::shared_ptr<ov::Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                          ov::element::Type input_type,
                                          ov::Shape input_shape,
                                          ov::Shape constant_shape,
                                          size_t binary_transpose_input_idx) {
    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, constant_shape, ov::Shape{1});

    NodePtr binary_op;
    if (!binary_transpose_input_idx)
        binary_op = binary_factory->create(X, in_constant);
    else
        binary_op = binary_factory->create(in_constant, X);

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(binary_op, ng_order0);

    return std::make_shared<ov::Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                                   ov::element::Type input_type,
                                                   ov::Shape input_shape,
                                                   ov::Shape constant_shape,
                                                   size_t binary_transpose_input_idx) {
    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

    auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, constant_shape, ov::Shape{1});

    std::vector<size_t> dims(input_shape.size() - constant_shape.size());
    std::iota(dims.begin(), dims.end(), 0);
    auto unsqueeze_const = std::make_shared<ov::opset9::Constant>(ov::element::i64, ov::Shape{dims.size()}, dims);
    auto unsqeeze = std::make_shared<ov::opset9::Unsqueeze>(in_constant, unsqueeze_const);

    auto ng_order1 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose1 = std::make_shared<ov::opset9::Transpose>(unsqeeze, ng_order1);

    NodePtr binary_op;
    if (!binary_transpose_input_idx)
        binary_op = binary_factory->create(transpose0, transpose1);
    else
        binary_op = binary_factory->create(transpose1, transpose0);

    return std::make_shared<ov::Model>(ov::OutputVector{binary_op}, ov::ParameterVector{X});
}

std::vector<ov::Shape> constant_shapes = {ov::Shape{96, 55, 55}, ov::Shape{1}};

}  // namespace incompat_shapes
}  // namespace backward

namespace forward {
namespace incompat_shapes {

std::shared_ptr<ov::Model> CreateFunction(BinaryFactoryPtr binary_factory,
                                          ov::element::Type input_type,
                                          ov::Shape input_shape,
                                          ov::Shape constant_shape,
                                          size_t binary_transpose_input_idx) {
    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, constant_shape, ov::Shape{1});

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(X, ng_order0);

    NodePtr binary_op;
    if (!binary_transpose_input_idx)
        binary_op = binary_factory->create(transpose0, in_constant);
    else
        binary_op = binary_factory->create(in_constant, transpose0);

    return std::make_shared<ov::Model>(ov::OutputVector{binary_op}, ov::ParameterVector{X});
}

std::shared_ptr<ov::Model> CreateReferenceFunction(BinaryFactoryPtr binary_factory,
                                                   ov::element::Type input_type,
                                                   ov::Shape input_shape,
                                                   ov::Shape constant_shape,
                                                   size_t binary_transpose_input_idx) {
    auto X = std::make_shared<ov::opset9::Parameter>(input_type, input_shape);

    auto in_constant = std::make_shared<ov::opset9::Constant>(input_type, constant_shape, ov::Shape{1});

    std::vector<size_t> dims(input_shape.size() - constant_shape.size());
    std::iota(dims.begin(), dims.end(), 0);
    auto unsqueeze_const = std::make_shared<ov::opset9::Constant>(ov::element::i64, ov::Shape{dims.size()}, dims);
    auto unsqeeze = std::make_shared<ov::opset9::Unsqueeze>(in_constant, unsqueeze_const);

    auto ng_order1 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 3, 1, 2});
    auto transpose1 = std::make_shared<ov::opset9::Transpose>(unsqeeze, ng_order1);

    NodePtr binary_op;
    if (!binary_transpose_input_idx)
        binary_op = binary_factory->create(X, transpose1);
    else
        binary_op = binary_factory->create(transpose1, X);

    auto ng_order0 = std::make_shared<ov::opset9::Constant>(ov::element::u64, ov::Shape{4}, ov::Shape{0, 2, 3, 1});
    auto transpose0 = std::make_shared<ov::opset9::Transpose>(binary_op, ng_order0);

    return std::make_shared<ov::Model>(ov::OutputVector{transpose0}, ov::ParameterVector{X});
}

std::vector<ov::Shape> constant_shapes = {ov::Shape{55, 55, 96}, ov::Shape{1}};

}  // namespace incompat_shapes
}  // namespace forward

}  // namespace single_consumer
}  // namespace binary

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingBinaryIncompatShapesBackwardTestSuite,
    TransposeSinkingBinaryIncompatShapesTestFixture,
    ::testing::Combine(::testing::ValuesIn(binary_factories),
                       ::testing::Values(CREATE_PASS_FACTORY(TransposeSinkingBinaryElementwiseBackward)),
                       ::testing::Values(ov::Shape{1, 96, 55, 55}),
                       ::testing::ValuesIn(binary::single_consumer::backward::incompat_shapes::constant_shapes),
                       ::testing::Values(binary::single_consumer::backward::incompat_shapes::CreateFunction),
                       ::testing::Values(binary::single_consumer::backward::incompat_shapes::CreateReferenceFunction),
                       ::testing::Values(ov::element::f32),
                       ::testing::ValuesIn(binary_transpose_input_indexes)),
    TransposeSinkingBinaryIncompatShapesTestFixture::get_test_name);

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingBinaryIncompatShapesForwardTestSuite,
    TransposeSinkingBinaryIncompatShapesTestFixture,
    ::testing::Combine(::testing::ValuesIn(binary_factories),
                       ::testing::Values(CREATE_PASS_FACTORY(TransposeSinkingBinaryElementwiseForward)),
                       ::testing::Values(ov::Shape{1, 96, 55, 55}),
                       ::testing::ValuesIn(binary::single_consumer::forward::incompat_shapes::constant_shapes),
                       ::testing::Values(binary::single_consumer::forward::incompat_shapes::CreateFunction),
                       ::testing::Values(binary::single_consumer::forward::incompat_shapes::CreateReferenceFunction),
                       ::testing::Values(ov::element::f32),
                       ::testing::ValuesIn(binary_transpose_input_indexes)),
    TransposeSinkingBinaryIncompatShapesTestFixture::get_test_name);
