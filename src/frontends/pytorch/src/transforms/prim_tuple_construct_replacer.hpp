#pragma once

#include "openvino/frontend/pytorch/visibility.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

class PYTORCH_API DecomposeTupleResults : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ov::frontend::pytorch::pass::DecomposeTupleResults");
    bool run_on_model(const std::shared_ptr<Model>& model) override;
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
