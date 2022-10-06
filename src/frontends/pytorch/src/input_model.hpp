// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/pytorch/decoder.hpp"
#include "openvino/frontend/pytorch/frontend.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

class InputModel : public ov::frontend::InputModel {
    friend class FrontEnd;
    std::shared_ptr<Decoder> m_model;

public:
    explicit InputModel(std::shared_ptr<Decoder> model) : m_model(model) {}
    // TODO: pass telemetry extension to this ctor
};

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
