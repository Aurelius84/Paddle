/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/c_allreduce_op.h"

namespace paddle {
namespace framework {
class OpDesc;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
namespace platform {
struct CPUPlace;
struct float16;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

class CAllReduceProdOpMaker : public CAllReduceOpMaker {
 protected:
  std::string GetName() const override { return "Prod"; }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(c_allreduce_prod, ops::CAllReduceOp,
                             ops::CAllReduceProdOpMaker);

REGISTER_OP_CPU_KERNEL(c_allreduce_prod,
                       ops::CAllReduceOpCPUKernel<ops::kRedProd, float>,
                       ops::CAllReduceOpCPUKernel<ops::kRedProd, double>,
                       ops::CAllReduceOpCPUKernel<ops::kRedProd, int>,
                       ops::CAllReduceOpCPUKernel<ops::kRedProd, int64_t>,
                       ops::CAllReduceOpCPUKernel<ops::kRedProd, plat::float16>)
