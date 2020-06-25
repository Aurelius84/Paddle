// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <cstdint>
#include "paddle/fluid/framework/unroll_array_ops.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

template <typename T, size_t N>
/*
 * 为什么要自己实现一个Array类?
 * 1. 为了支持多设备？
 * 2. 函数均为inline内联
 */
class Array {
 public:
  static constexpr size_t kSize = N; // array的大小，声明为静态变量，有什么特殊作用么？

  HOSTDEVICE inline Array() {}

  template <typename... Args> // 多参数的模板
  HOSTDEVICE inline explicit Array(const T &val, Args... args) {
    static_assert(N == sizeof...(Args) + 1, "Invalid argument");
    UnrollVarArgsAssign<T>::Run(data_, val, args...);
  }

  HOSTDEVICE inline void Fill(const T &val) {
    UnrollFillConstant<N>::Run(data_, val);
  }

  HOSTDEVICE inline const T *Get() const { return data_; } // const对象调用的函数

  HOSTDEVICE inline T *GetMutable() { return data_; }  // const函数没有词此函数

  HOSTDEVICE inline T &operator[](size_t i) { return *advance(data_, i); }

  // Writing "return data_[i]" would cause compilation warning/error:
  // "array subscript is above array bound" in Python 35 CI.
  // It seems that it is a false warning of GCC if we do not check the bounds
  // of array index. But for better performance, we do not check in operator[]
  // like what is in STL. If users want to check the bounds, use at() instead
  HOSTDEVICE inline const T &operator[](size_t i) const {
    return *advance(data_, i);
  }

  HOSTDEVICE inline T &at(size_t i) {
#ifndef __CUDA_ARCH__
    PADDLE_ENFORCE_LT(
        i, N, platform::errors::OutOfRange("Array index out of bounds."));
#endif
    return (*this)[i];
  }

  HOSTDEVICE inline const T &at(size_t i) const {
#ifndef __CUDA_ARCH__
    PADDLE_ENFORCE_LT(i, N, "Array index out of bounds");
#endif
    return (*this)[i]; // 调用[]操作
  }

  HOSTDEVICE constexpr size_t size() const { return N; }

  // 若两个array里的元素值都是相当的，则返回true
  HOSTDEVICE inline bool operator==(const Array<T, N> &other) const {
    return UnrollCompare<N>::Run(data_, other.data_);
  }

  HOSTDEVICE inline bool operator!=(const Array<T, N> &other) const {
    return !(*this == other);
  }

 private:
  template <typename U>
  HOSTDEVICE static inline U *advance(U *ptr, size_t i) {
    return ptr + i;  // 支持data[i]操作，此操作不会做index越界检查，为了提高性能
  }

  T data_[N];
};

template <typename T>
class Array<T, 0> { // 特化当N为0时
 public:
  static constexpr size_t kSize = 0;

  HOSTDEVICE inline Array() {}

  HOSTDEVICE inline void Fill(const T &val) {}

  HOSTDEVICE inline constexpr T *Get() const { return nullptr; }

  // Add constexpr to GetMutable() cause warning in MAC
  HOSTDEVICE inline T *GetMutable() { return nullptr; }

  HOSTDEVICE inline T &operator[](size_t) {
#ifdef __CUDA_ARCH__
    static T obj();
    return obj;
#else
    PADDLE_THROW("Array<T, 0> has no element");
#endif
  }

  HOSTDEVICE inline const T &operator[](size_t) const {
#ifdef __CUDA_ARCH__
    static const T obj();
    return obj;
#else
    PADDLE_THROW("Array<T, 0> has no element");
#endif
  }

  HOSTDEVICE inline T &at(size_t i) { return (*this)[i]; }

  HOSTDEVICE inline const T &at(size_t i) const { return (*this)[i]; }

  HOSTDEVICE constexpr size_t size() const { return 0; }

  HOSTDEVICE constexpr bool operator==(const Array<T, 0> &other) const {
    return true;
  }

  HOSTDEVICE constexpr bool operator!=(const Array<T, 0> &other) const {
    return false;
  }
};

}  // namespace framework
}  // namespace paddle
