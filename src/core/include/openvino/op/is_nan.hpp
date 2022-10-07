// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v10 {
/// \brief Boolean mask that maps NaN values to true and other values to false.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API IsNaN : public Op {
public:
    OPENVINO_OP("IsNaN", "opset10");
    /// \brief Constructs a isNaN operation.
    IsNaN() = default;

    IsNaN(const Output<Node>& data);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;
};
}  // namespace v10
}  // namespace op
}  // namespace ov
