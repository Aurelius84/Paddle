/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <cmath>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/search_compute_2.h"
namespace paddle {
namespace operators {
    using Tensor = framework::Tensor;
    using LoDTensor = framework::LoDTensor;
    using LoD = framework::LoD;

    class SearchFCOpMaker : public framework::OpProtoAndCheckerMaker {
    public:
        void Make() override {
            AddInput("X",
                     "X (Tensor, default Tensor<float>) Input variable which "
                     "should contain lod information.");
            AddInput("W", "W (Tensor)");
            AddInput("b", "b (Tensor)");
            AddAttr<int>("out_size", "out_size: the output size")
                    .SetDefault(0)
                    .EqualGreaterThan(1);
            AddOutput("Out", "Out (Tensor, default Tensor<float>) Output variable");
            AddComment(R"DOC(
  SearchFC

  NOTE: only support 'float32' data type now.
)DOC");
        }
    };
    class SearchFCOP : public framework::OperatorWithKernel {
    public:
        using framework::OperatorWithKernel::OperatorWithKernel;
        void InferShape(framework::InferShapeContext* ctx) const override {
            PADDLE_ENFORCE(ctx->HasInput("X"), "X(Input) should not be null.");
            PADDLE_ENFORCE(ctx->HasInput("W"), "W(Input) should not be null.");
            PADDLE_ENFORCE(ctx->HasInput("b"), "b(Input) should not be null.");
            PADDLE_ENFORCE(ctx->HasOutput("Out"), "Out(Output) should not be null.");
            auto x_dims = ctx->GetInputDim("X");
            PADDLE_ENFORCE_EQ(x_dims.size(), 2, "The rank of X(Input) should be 2.");
            auto w_dims = ctx->GetInputDim("W");
            PADDLE_ENFORCE_EQ(w_dims.size(), 2, "W should be 2-D tensor");
            auto b_dims = ctx->GetInputDim("b");
            PADDLE_ENFORCE_EQ(b_dims.size(), 1, "b should be 1-D tensor");
            int out_size = ctx->Attrs().Get<int>("out_size");
            ctx->SetOutputDim("Out", framework::make_ddim({-1, out_size}));
            if (ctx->IsRuntime()) {
                PADDLE_ENFORCE_EQ(w_dims[1], x_dims[1], "wrong shape: w_dims[1] != x_dims[1]");
            }
            else {
                // compile time
            }
        }
    };
    template <typename DeviceContext, typename T>
    class CPUSearchFCOPKernel : public framework::OpKernel<T> {
    public:
        void Compute(const framework::ExecutionContext& ctx) const override {
            auto* bottom = ctx.Input<Tensor>("X");
            auto* w = ctx.Input<Tensor>("W");
            auto* b = ctx.Input<Tensor>("b");
            auto* top = ctx.Output<Tensor>("Out");
            int out_size = ctx.Attr<int>("out_size");  // 100
            int batch = bottom->dims()[0];
            int _out = w->dims()[0];  // 100
            int _in = w->dims()[1];   // 228
            top->Resize(framework::make_ddim({bottom->dims()[0], out_size}));
            const auto* bottom_data = bottom->data<T>();
            auto* top_data = top->mutable_data<T>(ctx.GetPlace());
            const auto* weights = w->data<T>();
            auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);
            call_gemm(blas, CblasNoTrans, CblasTrans, batch, _out, _in, 1.0f,
                      bottom_data, weights, 0.0f, top_data);
            if (true) {
                const auto* bias_data = b->data<T>();
                for (int i = 0; i < batch; ++i) {
                    // add bias here
                    sse_eltadd(top_data + i * _out, bias_data, top_data + i * _out, _out);
                }
            }
        }
    };
    class SearchFCOpGrad : public framework::OperatorWithKernel {
    public:
        using framework::OperatorWithKernel::OperatorWithKernel;
        void InferShape(framework::InferShapeContext* ctx) const override {
            PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
            PADDLE_ENFORCE(ctx->HasInput("W"), "Input(W) should not be null.");
            PADDLE_ENFORCE(ctx->HasInput("b"), "Input(b) should not be null.");
            PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                           "Input(Out@GRAD) of SequencePadGradOp should not be null.");
            if (ctx->HasOutput(framework::GradVarName("X"))) {
                ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
            }
            if (ctx->HasOutput(framework::GradVarName("W"))) {
                ctx->SetOutputDim(framework::GradVarName("W"), ctx->GetInputDim("W"));
            }
            if (ctx->HasOutput(framework::GradVarName("b"))) {
                ctx->SetOutputDim(framework::GradVarName("b"), ctx->GetInputDim("b"));
            }
        }
    };
    template <typename DeviceContext, typename T>
    class CPUSearchFCOPGradKernel : public framework::OpKernel<T> {
    public:
        void Compute(const framework::ExecutionContext& ctx) const override {
            auto* bottom = ctx.Input<Tensor>("X");
            auto* w = ctx.Input<Tensor>("W");
            int _out = w->dims()[0];  // 100
            int _in = w->dims()[1];   // 228
            auto* d_out = ctx.Input<Tensor>(framework::GradVarName("Out"));
            auto* d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
            auto* d_w = ctx.Output<Tensor>(framework::GradVarName("W"));
            int batch = bottom->dims()[0];
            const auto* top_diff = d_out->data<T>();
            const auto* bottom_data = bottom->data<T>();
            auto* bottom_diff = d_x->mutable_data<T>(ctx.GetPlace());
            const auto* weights = w->data<T>();
            auto* weights_diff = d_w->mutable_data<T>(ctx.GetPlace());
            auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);
            call_gemm(blas, CblasTrans, CblasNoTrans, _out, _in, batch, (T)1.0,
                      top_diff, bottom_data, (T)0.0, weights_diff);
            call_gemm(blas, CblasNoTrans, CblasNoTrans, batch, _in, _out, (T)1.0, top_diff,
                      weights, (T)0.0, bottom_diff);
            if (true) {
                auto* d_b = ctx.Output<Tensor>(framework::GradVarName("b"));
                auto* bias_diff = d_b->mutable_data<T>(ctx.GetPlace());
                memset(bias_diff, 0.0, _out * sizeof(T));
                for (int i = 0; i < batch; ++i) {
                    sse_eltadd(bias_diff, top_diff + i * _out, bias_diff, _out);
                }

            }
        }
    };
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
namespace plt = paddle::platform;
namespace frm = paddle::framework;
REGISTER_OPERATOR(search_fc, ops::SearchFCOP, ops::SearchFCOpMaker,
                  frm::DefaultGradOpMaker<paddle::framework::OpDesc, true>,
                  frm::DefaultGradOpMaker<paddle::imperative::OpBase, true>);
REGISTER_OPERATOR(search_fc_grad, ops::SearchFCOpGrad);
REGISTER_OP_CPU_KERNEL(search_fc,
                       ops::CPUSearchFCOPKernel<plt::CPUDeviceContext, float>
//     ops::CPUSearchFCOPKernel<plt::CPUDeviceContext,
//                                       double>
);
REGISTER_OP_CPU_KERNEL(
        search_fc_grad, ops::CPUSearchFCOPGradKernel<plt::CPUDeviceContext, float>
//     ops::CPUSearchFCOPGradKernel<plt::CPUDeviceContext,
//                                           double>
);

