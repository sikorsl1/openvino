// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>

#include "openvino/frontend/pytorch/node_context.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
using CreatorFunction = std::function<OutputVector(NodeContext&)>;

std::map<std::string, CreatorFunction> get_supported_ops();

}  // namespace paddle
}  // namespace frontend
}  // namespace ov
