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

#pragma once

#include <string>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;
using DDim = framework::DDim;

constexpr int64_t kNoPadding = -1;

inline int scan_mark(const int64_t *data, const int len, const int mark,
                     int *offset) {
  int j = 0;
  int each_len = 0;
  for (int i = 0; i < len; i++) {
    if (data[i] == mark) {
      offset[j] = each_len;
      j++;
      each_len = 0;
    } else {
      each_len += 1;
    }
  }
  // for last term
  offset[j] = each_len;
  j++;
  return j;
}

template <typename T>
class MixEmbedKernel : public framework::OpKernel<T> {
 public:
  void pooling_ff(const std::string pool_type, int *max_index, const int len,
                  const int64_t *in, T *out, const int out_offset,
                  const int emb_size) {
    if (pool_type == "max") {
      sse_max(in, out + out_offset * emb_size,
              max_index + out_offset * emb_size, len, emb_size);
    } else {
      T alpha = (T)1.0 / len;
      if (pool_type == "sum") {
        alpha = (T)1.0;
      }
      for (int i = 0; i < len; ++i) {
        lego_cpu_axpby(emb_size, alpha, in + i * emb_size, 1.0,
                       out + out_offset * emb_size);
      }
    }

    void Compute(const framework::ExecutionContext &context) const override {
      auto *ids_t = context.Input<LoDTensor>("Ids");      // int tensor
      auto *output_t = context.Output<LoDTensor>("Out");  // float tensor
      auto *_max_index = context.Output<LoDTensor>("max_index");
      auto *_mix_char_offset = context.Output<LoDTensor>("mix_char_offset");
      auto *_buffer = context.Output<LoDTensor>("buffer");
      auto *table_var = context.InputVar("W");
      const int mark_sqlit = context.Attr<int>("mark_idx");
      const auto _pool_type = context.Attr<std::string>("pool_type");
      int64_t padding_idx = context.Attr<int64_t>("padding_idx");

      auto ids_dims = ids_t->dims();
      auto _cap_l = ids_dims[0];
      auto _cap_e = ids_dims[1];
      auto offset = ids_t->lod()[0];
      std::vector<int> top_offset;
      top_offset.resize(offset.size());
      top_offset[0] = 0;

      const auto *bottom_data = ids_t->data<int64_t>();

      int mix_char_l = ids_t->numel();
      // if most of slots are empty,then count() may be less than offset.size()
      if (mix_char_l < offset.size()) {
        mix_char_l = offset.size();
      }
      _mix_char_offset.Resize({mix_char_l});
      int *mix_offset = _mix_char_offset.mutable_data();
      // int64_t *ids = const_cast<int64_t *>(ids_t->data<int64_t>());
      // int64_t ids_numel = ids_t->numel();

      const auto *weights = table_var->data();
      for (int i = 0; i < top_offset.size() - 1; ++i) {
        int w = offset[i + 1] - offset[i];
        if (w == 0) {
          // keep a zero embedding vector
          top_offset[i + 1] = top_offset[i] + 1;
          mix_offset[top_offset[i]] = 1;
        } else {
          top_offset[i + 1] =
              top_offset[i] + scan_mark(bottom_data + offset[i], w, mark_sqlit,
                                        mix_offset + top_offset[i]);
        }
      }

      int top_l = top_offset[top_offset.size() - 1];
      output_t->Resize({top_l, _cap_e});
      T *top_data = output_t->mutable_data<T>(context.GetPlace());
      memset(top_data, 0, _cap_e * top_l * sizeof(T));

      if (_pool_type == "max") {
        _max_index->Resize({top_l, _cap_e});
      }

      int *max_index = _max_index.mutable_data();
      for (int i = 0; i < offset.size() - 1; ++i) {
        int w = offset[i + 1] - offset[i];
        if (w > 0) {
          int top_j = 0;
          int sum_num = 0;
          // gen sub embedding seqs
          unsigned int top_offset_j = top_offset[i] + top_j;
          sum_num = mix_offset[top_offset_j];
          _buffer.Resize({1, sum_num, _cap_e});
          T *sub_data = _buffer.mutable_data<T>(context.GetPlace());
          int sub_j = 0;
          for (int j = 0; j < w; ++j) {
            unsigned int word_idx =
                static_cast<unsigned int>(bottom_data[offset[i] + j]);
            if (word_idx != mark_sqlit) {
              memcpy((void *)(sub_data + sub_j * _cap_e),
                     (void *)(weights + word_idx * _cap_e), _cap_e * sizeof(T));
              sub_j += 1;
              continue;
            }
            pooling_ff(_pool_type, max_index,  sum_num, sub_data, top_data,
                       top_offset_j, _cap_e);
            // move to next sub seqs
            top_j++;
            top_offset_j = top_offset[i] + top_j;
            sum_num = mix_offset[top_offset_j];
            _buffer.Resize({1, sum_num, _cap_e});
            sub_data = _buffer.mutable_data<T>(context.GetPlace());
            sub_j = 0;
          }
          // to handle the end group of char for each sentences
          pooling_ff(_pool_type, max_index, sum_num, sub_data, top_data,
                     top_offset[i] + top_j, _cap_e);
        }
      }
    }
  };

  template <typename T>
  class MixEmbedGradKernel : public framework::OpKernel<T> {
   public:
    void pooling_bp(const std::string pool_type, Tensor *_max_index,
                    Tensor *_max_bp_buffer, const int len, T *weights,
                    const T *top_diff, const T *sub_index, const int out_offset,
                    const int emb_size, const float mlr) {
      if (pool_type == "max") {
        _max_bp_buffer->Resize({1, len, emb_size});
        T *diff = _max_bp_buffer->mutable_data<T>();
        memset(diff, 0, len * emb_size * sizeof(T));
        const int *max_index = _max_index->data<int>() + out_offset * emb_size;
        for (int i = 0; i < emb_size; ++i) {
          const int word_idx = sub_index[max_index[i]];
          diff[max_index[i] * emb_size + i] =
              top_diff[out_offset * emb_size + i];
        }
        for (int i = 0; i < len; ++i) {
          const int word_idx = sub_index[i];
          lego_cpu_axpby(emb_size, mlr, diff + i * emb_size, 1.0,
                         weights + word_idx * emb_size);
        }
      } else {
        if (pool_type == "default") {
          mlr = mlr / len;
        }
        for (int i = 0; i < len; ++i) {
          const int word_idx = sub_index[i];
          lego_cpu_axpby(emb_size, mlr, top_diff + out_offset * emb_size, 1.0,
                         weights + word_idx * emb_size);
        }
      }
    }

    void Compute(const framework::ExecutionContext &context) const override {
      auto *ids_t = context.Input<LoDTensor>("Ids");  // int tensor
      auto *output_t = context.Input<LoDTensor>("Out");  // float tensor
      auto *d_out = ctx.Input<LoDTensor>(framework::GradVarName("Out"));  // float tensor
      auto *_max_index = context.Output<LoDTensor>("max_index"); 
      auto *_mix_char_offset = context.Input<LoDTensor>("mix_char_offset");
      auto *_buffer = context.Input<LoDTensor>("buffer");
      auto *table_var = context.InputVar("W");
      const int mark_sqlit = context.Attr<int>("mark_idx");
      const auto _pool_type = context.Attr<std::string>("pool_type");
      const float _lr = context.Attr<float>("lr");
      // const float _l1_reg = context.Attr<float>("l1_reg");
      // const float _weight_decay = context.Attr<float>("weight_decay");

      auto _cap_l = ids_t->dims()[0];
      auto _cap_e = ids_t->dims()[1];

      // special treat when length is zero
      if (_cap_l == 0) {
        LOG(WARNING) << "bp a zero length input in embedding layer";
        return;
      }
      // end of special treat when length is zero

      const auto offset = ids_t->lod()[0];
      const auto top_offset = output_t->lod()[0];

      const auto *top_data = d_out->data<T>();
      auto *top_diff = d_out->mutable_data<T>(context.GetPlace());

      const int64_t *bottom_data = ids_t->data();
      T *weights = table_var->mutable_data<T>(context.GetPlace());

      auto mlr = -1.0 * _lr;

      int *mix_offset = _mix_char_offset.mutable_data<T>(context.GetPlace());
      // if (_l1_reg > 1e-10) {  // L1
      //   for (int k = 0; k < top_data->numel(); k++) {
      //     T val = top_data[k];
      //     int sign = (T(0) < val) - (val < T(0));
      //     top_diff[k] += _l1_reg * sign;
      //   }
      // }

      // if (_weight_decay > 1e-10) {  // L2
      //   sse_axpy(top_data, top_diff, top_data->numel(), _weight_decay);
      // }
      Tensor *_max_bp_buffer = nullptr;
      for (int i = 0; i < offset.size() - 1; ++i) {
        int w = offset[i + 1] - offset[i];
        if (w > 0) {
          int top_j = 0;
          int sum_num = 0;
          // keep sub embedding indices
          unsigned int top_offset_j = top_offset[i] + top_j;
          sum_num = mix_offset[top_offset_j];
          _buffer.Resize({1, sum_num, 1});
          T *sub_data = _buffer.mutable_data<T>(context.GetPlace());
          int sub_j = 0;

          for (int j = 0; j < w; ++j) {
            unsigned int word_idx =
                static_cast<unsigned int>(bottom_data[offset[i] + j]);
            if (word_idx != mark_sqlit) {
              sub_data[sub_j] = word_idx;
              sub_j += 1;
              continue;
            }
            pooling_bp(_pool_type, _max_index, _max_bp_buffer, sum_num, weights,
                       top_diff, sub_data, top_offset_j, _cap_e, mlr);
            // move to next sub seqs
            top_j++;
            top_offset_j = top_offset[i] + top_j;
            sum_num = mix_offset[top_offset_j];
            _buffer.Reisize({1, sum_num, 1});
            sub_data = _buffer.mutable_data<T>(context.GetPlace());
            sub_j = 0;
          }
          // to handle the group at end
          pooling_bp(_pool_type, _max_index, _max_bp_buffer, sum_num, weights,
                     top_diff, sub_data, top_offset_j, _cap_e, mlr);
        } else {
          DLOG("zero len sequence %d / %d", i, top_offset.size() - 1);
        }
      }
    }
  };

}  // namespace operators
}  // namespace paddle
