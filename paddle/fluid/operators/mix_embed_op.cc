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

#include "paddle/fluid/operators/mix_embed_op.h"

#include <memory>

#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"
#include "paddle/fluid/framework/var_type_inference.h"

namespace paddle {
namespace operators {

constexpr int64_t kNoPadding = -1;

class MixEmbedOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("W"), true,
                      "Input(W) of LookupTableOp should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("Ids"), true,
                      "Input(Ids) of LookupTableOp should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      "Output(Out) of LookupTableOp should not be null.");

    auto table_dims = ctx->GetInputDim("W");
    auto ids_dims = ctx->GetInputDim("Ids");
    int ids_rank = ids_dims.size();
    VLOG(5) << "ids rank is " << ids_rank << std::endl;
    PADDLE_ENFORCE_EQ(table_dims.size(), 2);
    PADDLE_ENFORCE_EQ(ids_dims[ids_rank - 1], 1,
                      "The last dimension of the 'Ids' tensor must be 1.");

    // in compile time, the lod level of ids must be 1
    framework::VarDesc* ids_desc =
        boost::get<framework::VarDesc*>(ctx->GetInputVarPtrs("Ids")[0]);
    PADDLE_ENFORCE_EQ(ids_desc->GetLoDLevel(), 1);

    ctx->SetOutputDim("Out", framework::make_ddim({-1, table_dims[1]}));

    if (ctx->GetOutputsVarType("Out")[0] ==
        framework::proto::VarType::LOD_TENSOR) {
      ctx->ShareLoD("Ids", /*->*/ "Out");
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("W"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class MixEmbedOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("W",
             "(Tensor) The input represents embedding tensors, "
             "which is a learnable parameter.");
    AddInput("Ids",
             "An input with type int32 or int64 "
             "contains the ids to be looked up in W. "
             "The last dimension size must be 1.");
    AddOutput("Out", "The lookup results, which have the same type as W.");
    AddOutput("max_index", "The max index of input");
    AddOutput("mix_char_offset", "The mix_char_offset");
    AddOutput("buffer", "buffer");
    AddAttr<std::string>("pool_type",
                         "(string, default sum) "
                         "A string specifying the reduction op. Currently sum "
                         "are supported, sum computes the weighted sum of the "
                         "embedding results for each row.")
        .SetDefault("sum");
    AddAttr<int>("mark_idx",
                 "(int64, default -1) "
                 "Separated mark to split Ids into multi sequences.")
        .SetDefault(kNoPadding);
    AddAttr<float>("lr", "learning rate.");
    // AddAttr<float>("l1_reg", "L1 regularzation");
    // AddAttr<float>("weight_decay", "weight decay");

    AddComment(R"DOC(
MixEmbeddingOperator.

Computes embeddings for the given ids and weights.
This operator is used to perform lookups on the parameter W,
then computes the weighted sum of the lookups results for each row
and concatenated into a dense tensor.
The input Ids should carry the LoD (Level of Details) information.
And the output will change the LoD information with input Ids.

)DOC");
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(MixEmbedGradOpNoBuffer, "W");

class MixEmbedGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());

    op->SetType("mix_embed_grad");

    op->SetInput("W", Input("W"));
    op->SetInput("Ids", Input("Ids"));
    op->SetInput("mix_char_offset", Input("mix_char_offset"));
    op->SetInput("max_index", Input("max_index"));
    op->SetInput("buffer", Input("buffer"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));

    op->SetOutput(framework::GradVarName("W"), InputGrad("W"));

    op->SetAttrMap(Attrs());
    return op;
  }
};

class MixEmbedOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto table_dims = ctx->GetInputDim("W");
    ctx->SetOutputDim(framework::GradVarName("W"), table_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(
        ctx.InputVar(framework::GradVarName("Out")));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class MixEmbedOpGradVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    auto out_var_name = ctx->Output(framework::GradVarName("W")).front();
    ctx->SetDataType(out_var_name, ctx->GetDataType(ctx->Input("W")[0]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(mix_embed, ops::MixEmbedOp, ops::MixEmbedOpMaker,
                  ops::MixEmbedGradOpDescMaker);

REGISTER_OPERATOR(mix_embed_grad, ops::MixEmbedOpGrad,
                  ops::MixEmbedGradOpNoBuffer,
                  ops::MixEmbedOpGradVarTypeInference);

REGISTER_OP_CPU_KERNEL(mix_embed, ops::MixEmbedKernel<float>,
                       //  ops::MixEmbedKernel<double>
);
REGISTER_OP_CPU_KERNEL(mix_embed_grad, ops::MixEmbedGradKernel<float>,
                      //  ops::MixEmbedGradKernel<double>
                       );
