// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/logical_kernel.h"

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/logical_functor.h"

namespace phi {

#define DEFINE_LOGICAL_BINARY_KERNEL(type)                                     \
  template <typename T, typename Context>                                      \
  void Logical##type##OpKernel(const Context& dev_ctx,                         \
                               const DenseTensor& x,                           \
                               const DenseTensor& y,                           \
                               DenseTensor* out) {                             \
    auto* out_ptr = dev_ctx.template Alloc<bool>(out);                         \
    bool need_broad_cast =                                                     \
        funcs::BroadCastInputs<context, T>(dev_ctx,                            \
                                           const_cast<DenseTensor&>(x),        \
                                           const_cast<DenseTensor&> y,         \
                                           out);                               \
    int ret = XPU_SUCCESS;                                                     \
    ret = xpu::logical_or<bool>(                                               \
        dev_ctx.x_context(), x.data<T>(), y.data<T>(), out_ptr, out->numel()); \
    PADDLE_ENFORCE_EQ(ret,                                                     \
                      XPU_SUCCESS,                                             \
                      errors::External("XPU API return wrong value[%d %s] in " \
                                       "op_name[logical_%s].",                 \
                                       ret,                                    \
                                       XPUAPIErrorMsg[ret],                    \
                                       #type));                                \
    if (need_broad_cast && dev_ctx.x_context()->xpu_stream != nullptr) {       \
      xpu_wait();                                                              \
    }                                                                          \
  }

DEFINE_LOGICAL_BINARY_KERNEL(And)
DEFINE_LOGICAL_BINARY_KERNEL(Or)
#undef DEFINE_LOGICAL_BINARY_KERNEL

template <typename T, typename Context>
void LogicalNotOpKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        DenseTensor* out) {
  if (x->numel() == 0) {
    return;
  }
  dev_ctx.template Alloc<bool>(out);
  int ret = xpu::logical_not<bool>(
      dev_ctx.x_context(), x.data<T>(), out->data<T>(), x.numel());

  PADDLE_ENFORCE_EQ(
      ret,
      XPU_SUCCESS,
      errors::External(
          "XPU API return wrong value[%d %s].", ret, XPUAPIErrorMsg[ret]));
}

}  // namespace phi

#define REGISTER_LOGICAL_CUDA_KERNEL(logical_and, func_type) \
  PD_REGISTER_KERNEL(logical_and,                            \
                     XPU,                                    \
                     ALL_LAYOUT,                             \
                     phi::Logical##func_type##OpKernel,      \
                     float,                                  \
                     double,                                 \
                     bool,                                   \
                     int64_t,                                \
                     int,                                    \
                     int8_t,                                 \
                     int16_t) {}

REGISTER_LOGICAL_CUDA_KERNEL(logical_and, And)
REGISTER_LOGICAL_CUDA_KERNEL(logical_or, Or)
REGISTER_LOGICAL_CUDA_KERNEL(logical_not, Not)
