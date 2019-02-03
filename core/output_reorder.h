// ======================================================================== //
// Copyright 2009-2019 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "node.h"
#include "image.h"

namespace oidn {

  // Output reorder node
  template<int K, class TransferFunc>
  class OutputReorderNode : public Node
  {
  private:
    std::shared_ptr<memory> src;
    const float* srcPtr;
    int H1;
    int W1;

    Image output;

    std::shared_ptr<TransferFunc> transferFunc;

  public:
    OutputReorderNode(const std::shared_ptr<memory>& src,
                      const Image& output,
                      const std::shared_ptr<TransferFunc>& transferFunc)
      : src(src),
        output(output),
        transferFunc(transferFunc)
    {
      memory::primitive_desc srcPrimDesc = src->get_primitive_desc();
      const mkldnn_memory_desc_t& srcDesc = srcPrimDesc.desc().data;
      MAYBE_UNUSED(srcDesc);
      assert(srcDesc.format == BlockedFormat<K>::nChwKc);
      assert(srcDesc.ndims == 4);
      assert(srcDesc.data_type == memory::data_type::f32);
      assert(srcDesc.dims[0] == 1);
      // We assume output data is <= K OC
      assert(srcDesc.dims[1] == K);

      assert(output.height <= srcDesc.dims[2]);
      assert(output.width  <= srcDesc.dims[3]);

      srcPtr = (float*)src->get_data_handle();
      H1 = srcDesc.dims[2];
      W1 = srcDesc.dims[3];
    }

    void execute() override
    {
      const int C1 = K;
      const int H2 = output.height;
      const int W2 = output.width;

      //parallel_nd(H2, [&](int h)
      for (int h = 0; h < H2; h++)
      {
        for (int w = 0; w < W2; ++w)
        {
          float* dstPtr_C = (float*)output.get(h, w);

          // Source is in nChwKc format. In this case C is 1 so this is really nhwc
          const float* srcPtr_C = srcPtr + h*W1*C1 + w*C1;

          #pragma unroll
          for (int i = 0; i < 3; ++i)
          {
            // Load the value
            float x = srcPtr_C[i];

            // The CNN output may contain negative values or even NaNs, so it must be sanitized
            x = isfinite(x) ? max(x, 0.f) : 0.f;

            // Apply the inverse transfer function
            x = transferFunc->inverse(x);

            // Store the value
            dstPtr_C[i] = x;
          }
        }
      }
    }
  };

} // namespace oidn
