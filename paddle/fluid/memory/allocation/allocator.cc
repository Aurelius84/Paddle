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

#include "paddle/fluid/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

bool Allocator::IsAllocThreadSafe() const { return false; }

void Allocator::FreeImpl(Allocation* allocation) {
  Allocator* allocator = allocation->TopDecoratedAllocator();
  // 返回最外层的分配器指针，调用最外层的free
  allocator->Free(allocation);
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
